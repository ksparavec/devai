# Review Fixes 2026-07

_Fix all 116 findings from the 2026-07-22 full-repository code review of `main`,
ordered by irreversibility rather than raw severity._

## Status

**Executed 2026-07-23**, in **three** passes over the working tree. It
took three, not one: each adversarial review of the previous pass found
defects that pass had itself introduced, so the count is a fact about
the remediation, not a formality.

1. **Remediation pass.** Eleven parallel units, split by file ownership,
   worked the phases below.
2. **Adversarial-review repair pass.** An adversarial review of pass 1
   found regressions and half-fixes in its own output; eleven units then
   repaired those. Several pass-1 changes were narrowed or reverted in
   pass 2 (for example the store-gap gate that turned the read-only
   `make model-fit` diagnostic into a fatal error, and a `RECREATE=1`
   path that could delete data without a `WIPE=1` opt-in).
3. **Final adversarial pass.** A third review found defects introduced
   by pass 2 and gaps pass 2 had left open. The substantive ones:

   - **Worker drain was latched.** Pass 2 made `draining` a state the
     worker never left, so a bounded drain permanently removed the
     worker from head-side routing. Now `draining -> ready` on
     completion via compare-and-swap (a shutdown arriving mid-drain
     still wins and stays terminal), plus a head-side degraded routing
     pass that falls back to a draining -- never a shutting-down --
     worker instead of 503ing a single-worker fleet.
   - **Bounded SkyPilot teardown abandoned billing VMs.** Pass 2's
     attempt/age bound dropped a pending `sky down` past the bound,
     orphaning a cloud VM that is still being paid for. Replaced with
     retry-forever plus exponential backoff (10s -> 15 min cap, which
     doubles as the log rate limiter), identity-keyed pending entries
     (`MarkForTeardown(cluster, instance)`) that close the name-reuse
     hole the bound only partly covered, and a conflict guard that
     refuses `sky down` when a different live worker holds the name.
   - **`make cluster-status` leaked the bearer token into argv** and
     targeted an endpoint that is not published to the host. The token
     now goes to curl over a pipe (`-H @-`), and the base URL is the
     overridable `DEVAI_HEAD_STATUS_URL` with a failure message naming
     the unpublished-port cause.
   - **The SGLang store-gap abort was unreachable on the unattended
     path.** It sat after `main()`'s `--name` short-circuit, so
     `make model-sync` never saw it. Now called from both paths, fatal
     on an enumerating `--download` run and on `--name X --download`
     only when `X` is itself a gap row.
   - **`setup-logs-volume.sh` still had two data-loss bypasses**: a
     foreign mount at the target was unmounted and its `/etc/fstab`
     entry rewritten before the guard fired, and `RECREATE=1`
     lvremoved an existing-but-unmounted LV with no `WIPE=1` opt-in.
     Both are closed.
   - **`devai-model-status` disagreed with `make model-fit`.** Added
     the `max_context` clamp both Python readers apply, gave Ollama
     exact-cell-only semantics (no winner-cell fallback), and gated
     vLLM/SGLang rows on the weights actually being on disk.

The three open questions below are resolved; see each for the decision
actually taken. What is **not** done is the live-GPU verification this
plan itself demands -- see "Unverified" after the questions.

Source review: `.claude/PRPs/reviews/local-2026-07-22-review.md` (untracked),
against `main` @ `57c4052`. 116 findings across 100 distinct sites; every
finding is assigned to exactly one phase below, verified programmatically.

## Dependencies

None. Every phase is cut from `main` as it stands.

Note: the uncommitted work in the `fix/review-findings` worktree is
deliberately **not** a dependency. That branch sits 16 commits behind `main`
and predates the per-model KV-cache-dtype work, so its diffs will not apply
cleanly. Treat it as prior art for wording, not as a starting point. If it is
rebased and landed first, re-check each phase below against the result -- it
already covers parts of Phase 1, 4, 5, 6 and 8.

## Enables / Unblocks

- Removes the two host-root-equivalent / credential-bearing network exposures
  that currently block any deployment outside a trusted LAN.
- Makes `make bench-report`, the picker's CODE% column, and the
  `devai-model-status` MCP tool agree with each other, which is a precondition
  for trusting model selection at all.
- Unblocks honest advertisement of cluster mode
  ([Plan: gpu-arbiter-cluster-mode](./gpu-arbiter-cluster-mode.md)), whose
  head-mode request serving was a 503 placeholder at `57c4052`. Resolved
  by open question 1 below: it is now wired to the single-host chain.
- Removes the probe/bench cache corruption paths that
  [Plan: bench-rewrite](./bench-rewrite.md) and
  [Plan: model-lifecycle-ledger](./model-lifecycle-ledger.md) both build on.

## Out of scope

- Any behaviour change not traceable to a review finding. This plan is
  remediation, not redesign.
- Re-running the review. The finding set is fixed as of `57c4052`.
- The AMD/ROCm verification gap in [docs/gpu-vendors.md](../gpu-vendors.md).
  Phase 9 corrects the `DEVAI_GPU_DEVICE` documentation, but actually
  verifying ROCm needs hardware and is its own work.
- Implementing head-mode cluster serving *as a feature*. Phase 6 either wires
  the existing chain in or corrects the docs (it wired it in -- see open
  question 1); a full cluster-mode Phase 3 belongs to its own plan.

## Open questions -- RESOLVED

1. **Phase 6, head-mode serving.** Wire `/v1/cluster/inbound` into the
   single-host chain, or correct `CLAUDE.md` and
   [docs/cluster-mode.md](../cluster-mode.md) to say it is unimplemented?
   The current state -- advertised as shipped, returns 503 -- is the only
   option that is not acceptable. Cost differs by roughly a day.

   **RESOLVED: wired in**, not documented away. The worker mounts
   `/v1/cluster/inbound` behind the bearer-token middleware and replays
   the forwarded body through its own single-host request handler, so a
   clustered request is rewritten exactly like a local one. Backend
   comes from `X-Devai-Backend` when the head sets it (it always does);
   the header-absent fallback ranks the worker's backends by probe
   evidence and answers 400 -- not 503 -- when nothing serves the name.
   `tests/test-cluster-preflight.sh` now explicitly FAILS if it sees
   the old 503 `not_implemented` placeholder.

2. **Phase 4, mixed-KV Ollama recreate policy** (`main.go:2528`). A pinned
   `@<ctx>` client plus a bare-name client currently recreate the container on
   every alternating request. Options: accept the thrash, refuse the bare name
   once a tier is pinned, or serve both from the larger tier. Needs an owner
   decision; it is a UX call, not a correctness one.

   **RESOLVED: an explicit `@<ctx>` pin is authoritative; a BARE name is
   served from the already-loaded tier without a recreate.** Only
   `ctxPinned` can move an Ollama tier, and then only when the pinned
   tier differs from the running one. The router does not re-derive a
   tier per request for a bare name, so the alternating-client thrash
   is gone and the picker's `@<ctx>` for mixed-KV models still selects
   the probed KV dtype exactly. Documented in
   [docs/router.md](../router.md) ("Ollama: `@<ctx>` is what pins a
   tier") and in the `devai-router` paragraph of `CLAUDE.md`.

3. **Phase 5, poisoned Ollama bench rows.** Delete the mislabelled multi-ctx
   rows, or re-bench them? Deleting is free and honest; re-benching costs GPU
   hours. Recommend delete, and let the next `make bench` refill.

   **RESOLVED: deleted** -- 4 rows across 2 models
   (`gemma4:26b-a4b-it-q4_K_M` at 131072 + 262144,
   `qwen3.6:35b-a3b-mtp-q4_K_M` at 65536 + 131072). The next `make bench`
   refills them with correctly-keyed `(model, backend, ctx)` rows. Not
   reversible by `git revert` (`deploy/.bench-cache.json` is gitignored), so
   all five host-local caches were snapshotted first to
   `~/.devai/backups/pre-review-fixes-20260723-094343.tar.gz`; the deletion
   itself was a tmp+rename rewrite that preserved `_meta` and the four
   single-ctx Ollama rows.

   Note for whoever re-benches: the deleted `qwen3.6:35b-a3b-mtp-q4_K_M` rows
   carried GPQA 0.8667 at 65K and 0.75 at 131K -- the measurement behind the
   per-tier q8_0 KV decision. The two rows came from separate bench runs two
   days apart rather than one `--all-ctx` sweep, and the q8_0-at-128K /
   f16-below split gives the delta a plausible physical cause, so the
   conclusion is probably sound -- but these rows could not *prove* it, which
   is why they went. Re-benching that model restores an auditable per-tier
   record.

## Unverified

The plan requires live-GPU test output before Phase 3 (router hot path)
may merge. That did not run:

- `make test-router` -- not run.
- `make test-vllm` -- not run.

Both need the GPU free and `make cache-down`, which was not available on
the execution host. Every claim about router and backend behaviour above
therefore rests on source reading and on the Go/Python unit tests, not on
a live serving round-trip. Phase 3 is **not** cleared to merge on this
evidence alone. This is still true after pass 3 -- neither test ran in
any of the three passes.

Two further things nothing in this remediation verified end to end:

- The cluster-mode drain/degraded-route behaviour and the SkyPilot
  teardown retry are covered by Go unit tests only. There is no live
  head + worker + cloud VM run behind them, and `MarkForTeardown` still
  has no production caller.
- The `devai-model-status` weights-on-disk gate is inert in the
  published container, which mounts no weight volumes -- it degrades to
  the un-gated verdict and says so in its `notes`. It was exercised on
  the host, not through the gateway.

## Context

The review ran 20 reviewer dimensions plus 3 completeness critics, with every
finding adversarially verified by 3 independent agents. 118 dimension findings
plus 16 sweep candidates were reduced to 116 confirmed. All 30 CRITICAL/HIGH
and 71 of the 78 MEDIUM/LOW sites were additionally hand-verified against
current source; none was refuted.

Two facts shape the ordering:

- **The full test suite passes on `main` today with all 116 findings present**
  -- 385 Python tests, both Go modules green under `build`, `vet` and `test`.
  Passing tests is therefore not evidence that a phase is safe; the router
  phase in particular needs live GPU tests.
- **Several findings are irreversible.** A wiped 300 GB weight store, a
  truncated sops secret, or a poisoned probe cache costs hours or days to
  rebuild. A LAN-reachable unauthenticated API costs nothing until it is
  exploited, but then costs everything. Both classes outrank ordinary bugs.

## Approach

Ordered by **irreversibility, then blast radius, then cost of delay** -- not
by severity label. A MEDIUM that destroys a six-hour probe cache outranks a
HIGH that needs a hostile LAN peer, so several MEDIUMs sit above HIGHs.

One branch per phase, cut from `main`, merged in order. Phases 1 and 2 can
share a single same-day PR. Phase 3 gets its own PR with live GPU test output
in the description.

| Phase | Theme                        | Sites | Findings | Est.      | Risk |
| ----- | ---------------------------- | ----- | -------- | --------- | ---- |
| 1     | Network exposure             | 6     | 6        | 30 min    | none |
| 2     | Irreversible data loss       | 11    | 11       | half day  | low  |
| 3     | Router hot path              | 14    | 22       | 1-2 days  | high |
| 4     | Silently wrong numbers       | 10    | 13       | 1 day     | med  |
| 5     | Cross-component contracts    | 10    | 10       | 1 day     | med  |
| 6     | Cluster + SkyPilot           | 8     | 10       | 1 day     | med  |
| 7     | Probe / bench robustness     | 11    | 12       | 1 day     | low  |
| 8     | Tooling, picker, backup      | 14    | 15       | half day  | low  |
| 9     | Documentation truth          | 16    | 17       | half day  | none |
|       | **Total**                    | 100   | 116      | ~7-8 days |      |

Phases 1 + 2 together are about one day and clear every irreversible risk. If
the budget is a single day, do those two and stop.

## Phase 1 -- Network exposure

Six edits in `deploy/docker-compose.yaml` and one in `cluster_head.go`. No
code paths change: every documented consumer reaches these services over
`devai-net` by service name, not through the published host port.

| Site                         | Change                                                                       |
| ---------------------------- | ---------------------------------------------------------------------------- |
| `docker-compose.yaml:245`    | `"127.0.0.1:${SKYPILOT_API_PORT:-46580}:46580"`                              |
| `docker-compose.yaml:256`    | replace `${HOME}:/root:rw` with `.aws:ro`, `.config/gcloud:ro`, `.config/sky` |
| `docker-compose.yaml:205-206`| `"127.0.0.1:${MCP_PORT:-8088}:8088"`                                         |
| `docker-compose.yaml:18`     | loopback-bind registry `5000` and apt `3142`                                  |
| `cluster_head.go:124`        | wrap `/v1/cluster/status` in `h.Token.AuthMiddleware`                         |

Rationale: the SkyPilot API server accepts anonymous `POST /launch`, `/exec`
and `/down` against real cloud credentials, and the MCP gateway holds a
**read-write Podman socket**, which is container-create, which is host root.
Both are reachable from any LAN host today.

The Podman socket cannot be made `:ro` -- the gateway must create containers.
Loopback binding is the mitigation. If the gateway must ever be reachable
off-host it needs an auth proxy in front, not a wider bind.

**Verify:** `podman compose --profile mcp --profile cluster config` shows
`127.0.0.1` on every published port; a `curl` from a second host is refused.

## Phase 2 -- Irreversible data loss

| Site                         | Fix                                                                                                                       |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `setup-logs-volume.sh:219`   | Gate the `find -delete` behind explicit `WIPE=1`; otherwise list contents and abort. Mirror `setup-libvirt-images-volume.sh`. **Shipped wider than scoped:** the `RECREATE=1` `rm -rf` at the top of the script destroys the same data and needed the same gate, so it got one -- `RECREATE=1` now aborts on a non-empty target (mounted or not) unless `WIPE=1` is set too. `make setup-logs` still does not forward `WIPE`, so that combination needs a direct script call. **Pass 3:** that guard was still bypassable two ways -- a target mounted from a FOREIGN device was unmounted and its `/etc/fstab` entry rewritten before the abort was reached, and `RECREATE=1` lvremoved an existing-but-unmounted LV (which reads as an empty directory) with no `WIPE=1` opt-in. Both closed: the foreign-mount refusal now runs before any LVM or fstab work on the normal path, and an existing `${VG}/${LV}` always counts as data-bearing. Net effect through make: `RECREATE=1 make setup-logs` refuses once the volume exists. |
| `Makefile:350`               | Download to a temp dir and swap on success. Today `rm -rf` runs first and the failure branch prints "existing cache preserved", which is false. |
| `render-secret.sh:66`        | Decrypt to `mktemp` in the same dir, `chmod 0600`, then `mv -f`. Today a failed decrypt truncates a good secret to zero bytes. |
| `render-secret.sh:41`        | Move the tmpfs check out from behind `[[ -d "$dst_dir" ]]` so it runs when the directory does not yet exist -- the exact case it exists for. |
| `setup-secrets-tmpfs.sh:64`  | Add `uid=`/`gid=` of `${SUDO_USER:-$USER}` to the mount options. Today every `*-secrets-render` target fails with EACCES.    |
| `_model_status.py:75`        | tmp+rename, matching every other cache writer in the tree.                                                                  |
| `model-sync.py:159`          | Run `cache-up` in a `try/finally` so a probe failure cannot leave the whole inference stack down.                            |
| `Makefile:1188`              | Bound the Ollama-readiness `until` loop with a timeout.                                                                     |
| `Makefile:1206`              | Add a cleanup `trap` to the `probe` recipe; an aborted probe leaves `devai-ollama` up with a VRAM-crippling `OLLAMA_GPU_OVERHEAD`. |
| `Makefile:1374`              | Drop the `; true` that swallows a `generate-catalog.py` crash under `REGEN=1`.                                              |
| `generate-catalog.py:597`    | Exit non-zero on upstream failure instead of writing a silently truncated `models.yaml`.                                     |

`setup-logs-volume.sh` is the sharpest: [CLAUDE.md](../../CLAUDE.md) instructs
operators to run it against `/var/cache/devai/vllm` with `SIZE=300G`, and the
natural ordering (pull weights, then give the path a volume) walks straight
into the `find -delete`.

**Verify:** `bash tests/test-backup-restore.sh`; run `setup-logs-volume.sh`
against a scratch directory containing a sentinel file and confirm it refuses.

## Phase 3 -- Router hot path

22 findings, 14 sites, all in `gpu-arbiter/main.go`. Highest regression risk in
the plan. Do the steps in order; each is independently testable.

1. **`main.go:2424` + `2419` (5 findings).** Move the `modelName == ""` guard
   above `a.stopOtherBackends(...)`. Today a bare `GET /` on port 11435 evicts
   the warm model and then 503s. Add a regression test asserting the other
   backend still holds the GPU.
2. **`main.go:2280` + `2303` (3 findings).** Give `unloadOllama`'s two HTTP
   calls a bounded client, and stop discarding the POST error and non-2xx
   status. Today a wedged Ollama daemon deadlocks the router while holding the
   arbiter mutex.
3. **`main.go:2346`.** Track container liveness separately from `bs.running`
   so `stopOtherBackends` also stops a backend whose launch failed but whose
   container is still alive on the GPU.
4. **`main.go:2320`.** `drainBackend` counts requests blocked on the very mutex
   it holds, guaranteeing a full `DRAIN_TIMEOUT` stall under load. Count only
   requests past the lock.
5. **`main.go:1471` + `1472`.** Key `modelContexts` / `modelSizes` by
   `(backend, name)`. The comment three lines below already documents this
   invariant for the neighbouring maps.
6. **`main.go:2486`.** Record the context the launch settled on
   (`lc.MaxContext`), not the requested one. The comment already claims this.
7. **`main.go:2747` + `2746` (4 findings).** Clamp a client-supplied
   `options.num_ctx` to the probed ceiling; today the clamp guards only the
   router's own value.
8. **`main.go:73` (2 findings).** Add an `envIntAllowZero` so
   `MAX_CONCURRENT_REQUESTS=0` means unlimited, as documented.
9. **`main.go:2528`.** Resolve open question 2 (mixed-KV recreate policy).
10. **`main.go:2764`.** Add the missing `setNumCtx` tests covering the
    `/api/chat` + `/api/generate` path gate the docs mark must-not-change.

**Verify:** `go test ./gpu-arbiter/`, `go vet`, then live `make test-router`
and `make test-vllm`. Do not merge this phase on unit tests alone.

## Phase 4 -- Silently wrong numbers

These never crash. They produce plausible numbers that are wrong, and every
model-selection decision downstream inherits the error.

1. **The `humaneval_` prefix collision -- 4 sites, one commit.**
   `bench_report.py:53` and `:155`, `benchcache.go:85`, `model-picker.py:468`
   all match bare `"humaneval_"`, which also matches `humaneval_plus_subset_*`;
   the max-`ran_at` tiebreak then picks HumanEval+. Use `"humaneval_subset_"`
   everywhere and add an explicit `humaneval_plus_subset_` field.
   `model-picker.py:435` already documents the correct prefix.
   **Fix all four together** -- a partial fix creates a new disagreement.
   **No re-bench needed**: the cached data is correct, only the readers are wrong.
2. **`_bench_core.py:687`.** `serving_alias_with_ctx` drops the ctx for Ollama,
   so `--ctx` / `--all-ctx` mints rows labelled 32K/64K/128K that were all
   served at one context. Honour the ctx for Ollama or reject the flag.
   **This one has poisoned data** -- see open question 3.
3. **`main.go:288`.** The Ollama prober writes `disable_verified: "error"` (a
   JSON string) into a Go `*bool`. One such entry fails the whole-file
   unmarshal and the router registers **zero** Ollama models; the sentinel is
   sticky, so re-probing never clears it. Fix the writer to leave the field
   absent on probe error, **and** grep existing caches for the string.
4. **`_bench_core.py:363`.** Stamp `host_env_id` per task, not per row; a
   forced re-bench of the default tasks currently re-labels surviving
   GPQA/MMLU results as having run under the new host environment.
5. **`bench_runner.py:886`.** The default task set omits `gpqa`, `mmlu_pro` and
   `humaneval_plus`, so the picker's default sort column and three of its seven
   benchmark columns stay blank after a plain `make bench`.
6. **`bench_runner.py:555`.** Widen the leak/latency `except` to match its
   siblings so one task's exception cannot abort a multi-model run.
7. **`humaneval.py:18`.** Implement the fork-bomb defence the comment claims
   (`RLIMIT_NPROC`) or delete the claim. This sandbox executes model-generated code.

**Verify:** `make bench-report` before and after against the existing cache --
the HumanEval column must change for every row carrying both tasks.

## Phase 5 -- Cross-component contracts

Each item is individually correct on both sides and broken only where they
meet. These are the findings a per-file review structurally cannot see.

| Site                              | Fix                                                                                                                                   |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `main.go:2054`                    | Key the recovery registry by `(backend, model)`. Today vLLM-only flags (`--language-model-only`, `--quantization modelopt`) and vLLM image pins are applied verbatim to SGLang launches. |
| `select-models.py:242`            | Nothing populates `SGLANG_MODELS_DIR`. **Live today:** the SGLang cache advertises 8 models with `fits=true` while `/var/cache/devai/sglang/` holds 0 files. Populate both stores, or fail loudly at launch. |
| `status.go:77`                    | `get_router_status` probes `host.containers.internal`, but the router publishes no host ports. Address `http://devai-router:11434` over `devai-net`. |
| `probecache.go:56` + `:15`        | `FitsAt` does exact-cell lookup against **single-cell** caches and ignores `serving_ok`. Accept any cell with ctx >= requested; honour `serving_ok` when present. |
| `docs/mcp-model-status.md:20`     | Correct the doc that enshrines the exact-cell rule. Fix alongside the above.                                                            |
| `select-models.py:221`            | Add a `source == "gguf"` branch to `is_downloaded()`; today every GGUF row re-downloads forever.                                        |
| `Makefile:1551`                   | Stage `deploy/models.yaml` in `install-systemd`, else compose bind-mounts a directory over it and the MTP registry loads empty.          |
| `benchcache.go:64` + `:66`        | Drop or explicitly deprecate the retired REAS/TOTAL composites; the doc claims parity with a picker formula that no longer exists.       |

**Verify:** call `list_fitting_models{vram_gb: 24, context: 32768, backend: "vllm"}`
and confirm the row count matches the picker at 32K.

## Phase 6 -- Cluster + SkyPilot

Features documented as shipped that are not. Lower urgency only because
cluster mode is opt-in; for anyone actually running a head, this is their
Phase 1.

1. **`cluster_main.go:145`.** Worker `/v1/cluster/inbound` is a hardcoded 503
   placeholder, so head mode cannot serve at all. Resolve open question 1.
   Note that `tests/test-cluster-preflight.sh:286` **asserts** the 503 -- that
   assertion must be inverted, not preserved.
2. **`cluster_head.go:311` (3 findings).** Apply `http.MaxBytesReader` (32 MiB)
   in `makeFrontendHandler`, `handleRegister` and `handleHeartbeat`, matching
   the guard `main.go:2664` already carries.
3. **`cluster_worker.go:285`.** Handle the head's 410 by re-registering. The
   head's own comment states this contract; the worker never implements it, so
   a head restart permanently orphans every worker.
4. **`skypilot_policy.go:269` + `:274`.** Drop a cluster from `pending` only
   when `tryDown` succeeded. The comment says "the next sweep will retry" while
   the code deletes the entry unconditionally -- **this orphans a billing cloud VM.**
5. **`cluster_proxy.go:37`.** `Timeout: 0` on head-to-worker forwarding; add a
   response-header timeout.
6. **`skypilot_policy.go:160`.** L40S is 48 GB, not 24.
7. **`cluster_auth_test.go:47`.** Test token rotation at the real 30s `CacheTTL`.

**Verify:** `make test-cluster-preflight`, after inverting the 503 assertion.

## Phase 7 -- Probe / bench robustness

| Site                                         | Fix                                                                                                                             |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `_probe_hf_common.py:123` + `:1174`          | No `finally` or signal handler anywhere: Ctrl-C leaks a GPU-holding container that silently contaminates later probe cells with false OOMs. Add guaranteed teardown. |
| `_probe_hf_common.py:1849` + `backends.md:195` | `PROBE_CONTEXTS` / `--ctx` is parsed, printed, then discarded -- `binary_search_max_ctx` never receives it. Honour it or remove it. |
| `_probe_load.py:531`                         | Read the cell's stamped `kv_cache_type` instead of the process-global dtype, so serving data matches what the cell advertises.     |
| `_probe_hf_common.py:1460`                   | Stop bumping `schema_version` without re-probing any cell.                                                                        |
| `_probe_hf_common.py:1915`                   | A model below the 32K floor gets no cell, no ledger entry and no output, and is silently retried forever. Record an exclusion.     |
| `probe-ollama-reasoning.py:345`              | Stamp `OLLAMA_FLASH_ATTENTION` into the cell; serve-time cannot otherwise reproduce the measured environment.                      |
| `probe-ollama-reasoning.py:957`              | Stale "(capped at X)" summary message.                                                                                            |
| `backend-flags.yaml:27` + `verify-backend-flags.py:60` | Pin `--kv-cache-dtype`, and replace the substring test with exact matching so a prefix-named flag cannot mask a removal.  |

## Phase 8 -- Tooling, picker, backup

| Site                                          | Fix                                                                                                                             |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `restore.go:114`                              | Reject a bare root-name archive entry. Today an entry named `deploy` resolves to the root itself, and restore renames the whole `deploy/` tree aside and writes a file over it, taking every probe/bench cache with it. |
| `restore.go:46`                               | Validate pre-existing symlinks at intermediate path components.                                                                   |
| `envfile.go:42`                               | Rewrite every duplicate `KEY=` line; `.env` is last-line-wins, so the stale later line currently wins.                            |
| `llmfit-catalog-diff.py:44` + `:134`          | Handle mapping-form `hf_repos` entries (`{'repo': ..., 'mtp': ...}`) -- `r.strip()` raises today, breaking `make catalog-suggest`. Guard the `:.2f` on a missing `gpu_vram_gb`. |
| `model-picker.py:2891`                        | `"@" in name` means vLLM, which mis-routes mixed-KV Ollama models pinned with `@<ctx>`. Decide from the probe cache, not the string. |
| `model-picker.py:2177`, `:3116`, `:2003`, `:2511` | Surface TOOLS in ranking; test mixed-KV pinning; stop scoring an unmeasured leak as perfect faithfulness while an unmeasured TPS scores zero; update the formula legend. |
| `bin/devai-agent:492`                         | Guard `int(pick["context"])`; it raises before preferences are persisted.                                                          |
| `_model_status.py:169`                        | `prune_to_catalog` is implemented and unit-tested but never called.                                                               |
| `docker-compose.yaml:220` + `mcp-gateway.env:3` | `--secrets=${MCP_SECRETS_PATH:-/dev/null}` never picks up the rendered file; hardcode `/secrets/.env`. Wire `env_file`, or delete the two `.env` files that claim compose reads them. |

## Phase 9 -- Documentation truth

17 findings of doc and comment drift. Cheap, and this repo treats docs as the
operator contract: a wrong env-var name silently ignores operator config.

- **Env vars:** `cluster-env.md:33` (`SKYPILOT_API_ENDPOINT` never read),
  `cluster-mode.md:128` (`DEVAI_HEAD_TOKEN_FILE` undocumented),
  `gpu-vendors.md:54` (`DEVAI_GPU_DEVICE` never reaches the router container),
  `worker-cloud-init.sh:31` (`DEVAI_WORKER_HOST` defaults to `localhost`, not
  `$(hostname)`).
- **Router docs:** `README.md:320` (`IDLE_TIMEOUT` default is 0, not 300),
  `router.md:281` (wrong injected reasoning field), `router.md:368` (the global
  `--kv-cache-dtype fp8` hardcode is gone), `backends.md:67`
  (`currentReasoningOverride` does not exist), `main.go:706` (imageStale comment).
- **CLAUDE.md:** `:36` stale counts and anchors; `:246` ten tracked `.md` files
  violate the ASCII-only rule that this very section states; `:322` the sops
  files are not gitignored (`.gitignore:128` un-ignores them); `:367`
  `reset_row_for_force` was deleted in `1e4e228`.
- **Other:** `cluster-mode.md:135` references two files that do not exist;
  `registries.conf:2` install instructions point at a nonexistent path;
  `speculative_test.go:227` is not gofmt-clean.

**Verify:** `gofmt -l gpu-arbiter/` is empty; an ASCII sweep over
`git ls-files '*.md'` is clean; every documented env var has a matching
`os.Getenv`.

## Combined risk register

| Risk                                                                 | Phase | Mitigation                                                                 |
| -------------------------------------------------------------------- | ----- | -------------------------------------------------------------------------- |
| Router regression under concurrency; unit tests will not catch it     | 3     | Own PR; live `make test-router` + `make test-vllm` before merge             |
| Partial `humaneval_` fix creates a new picker/report disagreement     | 4     | All four sites in one commit                                                |
| Poisoned Ollama bench rows survive the code fix                        | 4     | Explicit cache audit step; see open question 3                              |
| Sticky `"error"` sentinel already present in operator caches           | 4     | Grep existing caches as part of the fix, not after                          |
| Loopback binding breaks an undocumented remote consumer                | 1     | All documented consumers use `devai-net`; call out in release notes         |
| `WIPE=1` gate blocks an existing automation that relied on the wipe    | 2     | Grep the repo and ansible for callers before landing                        |
| Inverting the preflight 503 assertion masks a real cluster regression  | 6     | Invert only alongside the actual inbound wiring, never on its own           |

## Migration / rollback story

Every phase is a self-contained branch off `main` and reverts cleanly on its
own, with two exceptions that need care:

- **Phase 4 data remediation.** Deleting poisoned bench rows is not reversible
  by a git revert. Snapshot `deploy/.bench-cache.json` first -- for the
  2026-07-23 execution this was
  `~/.devai/backups/pre-review-fixes-20260723-094343.tar.gz`, which holds all
  five host-local caches (bench, the three probe caches, the model-status
  ledger) as they stood before any remediation.
- **Phase 2 `setup-secrets-tmpfs.sh`.** Changing the tmpfs ownership requires
  a remount, so `/run/devai` must be re-rendered after the change
  (`make secrets-tmpfs` then the relevant `*-secrets-render`).

No schema versions change in any phase. No probe or bench cache needs
regenerating except the Phase 4 remediation above; in particular the
`humaneval_` fix is read-side only and the existing measurements stay valid.

## Estimated effort

About 7-8 engineer-days total, split as in the Approach table. Phases 1 and 2
together are roughly one day and clear every irreversible risk; that is the
recommended minimum scope if the work has to be cut short.

## References

- Source review: `.claude/PRPs/reviews/local-2026-07-22-review.md` (untracked,
  local-only; regenerate with a full-repo review at `57c4052` if lost)
- Prior review, all findings resolved in `6cacc14`:
  `.claude/PRPs/reviews/local-2026-05-13-review.md`
- [Plan: gpu-arbiter-cluster-mode](./gpu-arbiter-cluster-mode.md) -- Phase 6 here
  corrects its head-mode serving gap
- [Plan: bench-rewrite](./bench-rewrite.md) -- Phase 4 here fixes readers of the
  schema it defines
- [Plan: model-lifecycle-ledger](./model-lifecycle-ledger.md) -- Phase 7 here
  fixes the exclusion-ledger gaps it introduced
- [docs/backends.md](../backends.md), [docs/router.md](../router.md) -- the
  invariants Phases 3 and 7 are measured against
