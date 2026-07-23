# Model lifecycle ledger + auto-sync

_Persist every "this model is unsuitable" verdict so unfit models are never
re-downloaded, re-probed, or re-listed -- and auto-onboard genuinely new
catalog rows (download + probe) in one closed loop._

## Status

In Progress (2026-06-23). All four open questions resolved (see "Decisions"
below). All 4 phases implemented + unit-tested; GPU-dependent confirmation
(the Gemma-4-31B-IT-NVFP4 scoped probe, a live `make model-sync` run) is
still pending on the host.

## Dependencies

- None. Builds entirely on existing scripts (`scripts/select-models.py`,
  `scripts/_probe_hf_common.py`, the three probe caches) and the catalog
  (`deploy/models.yaml`). No other plan is a prerequisite.

## Enables / Unblocks

- A hands-off model fleet: `make model-sync` keeps the on-disk set equal to
  "every catalog row that fits this host and loads," with no manual
  download/probe bookkeeping.
- Auditable exclusions: an operator can answer "why isn't gemma-4-31b-it
  served here?" from a single file instead of re-deriving the VRAM math.
- Quieter probe runs: `make probe-vllm` stops printing "not on disk" for
  rows that can never be downloaded on this host.

## Out of scope

- Changing the catalog itself (`deploy/models.yaml` stays the host-agnostic
  superset). The ledger is a host-LOCAL overlay, not a catalog edit.
- Extending the `recovery_image()` mechanism itself. Phase 1 USES it to
  register the existing vLLM "gemma" build for gemma-4 (decision 4), but the
  mechanism is unchanged -- no new per-model-image machinery is built here.
- Multi-host / cluster-mode fleet sync. The ledger is keyed to one host's
  GPU budget; a head-node fleet view is a later concern.
- Re-probing on a cadence / TTL. Exclusions persist until a model's weights
  change (new sha) or the operator clears them.

## Decisions (locked 2026-06-23)

1. **Ledger key = catalog `name` + `backend`.** Store `repo` and the last
   evaluated `sha` as fields. `name` survives a re-quant commit, so a
   verdict is not orphaned the way a `repo@sha` key would be (the gemma-4
   double-key bug). Ollama rows key by their `library:tag` name, which is
   also sha-stable.
2. **OOM is re-checked on a new sha; not sha-stable.** Only `too_big`,
   `too_small`, and `unsupported_arch` carry forward permanently. OOM is
   weight-specific -- a new/better quant of the same repo gets re-probed and
   may now fit.
3. **`model-sync` downloads AND probes in one invocation**, gated by a
   `SYNC_MAX_DOWNLOADS` budget so a large catalog diff cannot fill the disk
   unattended; `DRY_RUN=1` prints the plan without touching anything.
4. **gemma-4 is SERVED, not excluded -- on the DEFAULT image.** Correction
   to the original framing: the `vllm-openai:gemma-x86_64-cu130` build is for
   DiffusionGemma ONLY (already wired, and it regresses Qwen NVFP4 loading,
   so it must not touch regular gemma-4). The NVFP4 gemma-4 models serve on
   the default image with a recovery flag -- `Gemma-4-26B-A4B-NVFP4` already
   does (`--max-num-batched-tokens 4096`, the vision-encoder fix); Phase 1
   adds the same flag for `Gemma-4-31B-IT-NVFP4`. gemma-4-e2b-it's failure is
   that same multi-modal config issue (a flag fix), NOT unsupported_arch, so
   the classifier correctly leaves it non-terminal. The bf16 base rows
   (e2b/e4b too_small, 26b/31b too_big) are excluded by the size filter in
   Phase 3, not served. The classifier fix still lands as a general
   robustness improvement (root cause is now captured; a truly unsupported
   arch classifies terminally).

## Context

The model lifecycle today is three loosely-coupled steps -- `make model-pull`
(download), `make probe-*` (classify fit + capability), and the picker
(show only `fits=true`). Most "mark and skip" already works:
`scripts/select-models.py` excludes anything outside the VRAM window
`min_total <= total <= vram_budget` (`select-models.py:1232`, with
`min_total = vram * MIN_VRAM_FRACTION`, default 0.5 -- `:1338`, `:1409`),
and `_row_probe_rejected` (`:924`) drops any model whose probe set
`capability=error/unsupported_arch` or left a `fits=false` cell at the
target `(vram, ctx)`. So a probed-and-OOM'd or too-big/too-small model is
not re-downloaded.

Three real gaps remain, surfaced by the gemma-4 probe run:

1. **Mis-classification makes terminal failures look transient.**
   `classify_failure_logs` inspects only the last 30 log lines
   (`_probe_hf_common.py:757`); a genuine "unknown architecture
   Gemma4ForConditionalGeneration" failure whose root-cause line scrolled
   past matches no arch pattern and is tagged `infra`. And
   `refresh_top_level_from_cells` only marks a model terminally unsupported
   when `kind in {arch, quant}` (`:1344`); `infra` is explicitly
   non-terminal (`:1333`). Net: gemma-4 is treated as retryable forever.

2. **The cache is keyed `repo@sha`** (`:1568`), so a verdict does not
   survive a re-quant/commit. gemma-4-e2b-it already has two stranded keys
   (`@6b7e72c`, `@70af34e`) after a `catalog-regen`; the old verdict is
   orphaned and the model re-probes under the new key.

3. **Never-downloaded fit-exclusions are recomputed, never recorded.** The
   bf16 `google/gemma-4-31b-it` (~58 GB) is excluded by the VRAM window
   every run, but nothing writes that down, so it reappears as a
   "not on disk" skip line each probe and is not auditable.

And there is no closed loop: a genuinely new catalog row (e.g. a freshly
discovered `qwen3.7`) requires a manual `model-pull` then `probe-*`.

## Approach

Introduce one host-local **exclusion ledger** (`deploy/.model-status.json`,
gitignored like the probe caches) that records every "do not bother with
this model on this host" verdict with a reason, the `(host_vram, ctx)` it
was judged at, and sha-stability. The downloader and the probers consult it
to skip excluded rows entirely (no download, no probe, no "not on disk"
noise) and write to it when they reach an exclusion verdict. A new
`make model-sync` target diffs the regenerated catalog against
(ledger + probe caches), and for each genuinely new row runs the full
download -> probe -> record loop. Phase 1 (fix the classifier) and Phase 2
(make terminal verdicts sha-stable) are prerequisites so the ledger records
*correct*, *durable* verdicts; Phases 3-4 add the ledger and the loop.

---

## Phase 1 -- Failure classification fix

### Goal

A genuine architecture/quant load failure classifies as `arch`/`quant`
(terminal), not `infra` (transient), so it can be permanently excluded.
Genuine infrastructure flakiness stays transient.

### Deliverables

```
scripts/_probe_hf_common.py     modify -- classify_failure_logs scans the
                                          FULL log for arch/quant/oom
                                          patterns (keep 30-line display
                                          excerpt); extend _ARCH_ERROR_PATTERNS
tests/python/test_probe_*        modify -- arch-line-past-tail classifies arch
deploy/recovery-flags.json       modify -- add --max-num-batched-tokens 4096
                                          for Gemma-4-31B-IT-NVFP4 (mirror the
                                          26B; decision 4). NO image override --
                                          serves on the default vLLM image
```

### Detailed steps

1. Reproduce the gemma-4-e2b-it launch once and capture the full container
   log (the prober already grabs `tail=500` via `container_logs`; print or
   persist it) to read the actual root-cause string vLLM emits for an
   unsupported architecture.
2. In `classify_failure_logs` (`:747`), run the arch/quant pattern match
   against the WHOLE log (`lc`), not just the 30-line excerpt -- the excerpt
   stays for display only. (The function already lowercases the full `logs`;
   the bug is only that pattern misses lead nowhere because the tell-tale
   line is earlier than the persisted excerpt -- confirm and widen
   `_ARCH_ERROR_PATTERNS` to include the verified gemma string, e.g.
   "are not supported for now", "unknown architecture", "ValueError: Model
   architectures".)
3. Leave `infra` as the genuine-fallback bucket (container won't start, GPU
   absent, port clash) -- those SHOULD remain retryable.
4. Per decision 4, add `--max-num-batched-tokens 4096` for
   `Gemma-4-31B-IT-NVFP4` in `deploy/recovery-flags.json` (mirror the working
   `Gemma-4-26B-A4B-NVFP4` entry; the vision-encoder fix). No image override
   -- it serves on the default vLLM image. Confirm with
   `make probe-vllm PROBE_REPO=Gemma-4-31B-IT-NVFP4 PROBE_FORCE=1`. (The
   `gemma-x86_64-cu130` build stays reserved for DiffusionGemma, which is
   already wired and whose comment notes it regresses Qwen NVFP4 loading.)

### Exit criteria

- A vLLM launch that fails with an unsupported-architecture error records
  `kind=arch` and `refresh_top_level_from_cells` sets
  `capability=unsupported_arch`.
- `_row_probe_rejected` then drops the model from download candidates, and
  `run_probe_pass` skips it (`:1592`) -- verified by a unit test feeding a
  synthetic log whose arch line precedes the 30-line tail.

### Phase 1 risks

| Risk                                              | Mitigation                                              |
| ------------------------------------------------- | ------------------------------------------------------- |
| Over-broad arch pattern tags a transient as arch  | Patterns must be vLLM/SGLang load-error strings only; unit-test both directions |
| A model that only needs the gemma image gets excluded | Document the `recovery_image` opt-in; Phase 3 ledger reason makes it visible/clearable |

---

## Phase 2 -- Sha-stable terminal verdicts + orphan pruning

### Goal

A terminal verdict (`unsupported_arch`) survives a re-quant/commit instead
of orphaning under the old sha; stale `repo@sha` entries are pruned.

### Deliverables

```
scripts/_probe_hf_common.py     modify -- on ensure_entry for a new sha of a
                                          known repo, carry forward a terminal
                                          unsupported_arch verdict; add
                                          prune_orphaned_shas(cache, catalog)
scripts/probe-vllm-reasoning.py  modify -- call the prune at end of run_probe_pass
tests/python/test_probe_*        modify -- new-sha inherits unsupported_arch;
                                          orphan prune drops dead keys
```

### Detailed steps

1. In `ensure_entry` (`:1250`), when a new `repo@sha` is created and an older
   entry for the same `repo` carried `capability=unsupported_arch`, copy that
   terminal capability forward (arch does not change with a re-quant of the
   same repo). Do NOT carry `oom` (weight-specific -- re-check) per open
   question 2.
2. Add `prune_orphaned_shas(cache, catalog_rows)`: drop any `repo@sha` whose
   `repo` still exists in the catalog but at a different `sha`, after
   carrying forward any terminal verdict. Keep the newest sha.
3. Call it once at the end of `run_probe_pass` (guarded by
   `not args.no_cache_write`).

### Exit criteria

- After a simulated sha bump, the new key reports `unsupported_arch` without
  a re-probe, and the old key is gone.
- `make probe-vllm` no longer re-probes gemma-4 once Phase 1 marks it arch.

### Phase 2 risks

| Risk                                    | Mitigation                                          |
| --------------------------------------- | --------------------------------------------------- |
| Carrying forward a stale verdict hides a fix | Only `unsupported_arch` carries; operator can `PROBE_FORCE_ARCH=1` to re-evaluate |
| Pruning deletes a still-referenced sha  | Prune only when the same repo exists at a different sha in the current catalog |

---

## Phase 3 -- Exclusion ledger + gating

### Goal

One host-local file records every exclusion (including never-downloaded
fit-exclusions) with a reason; the downloader and probers consult it to skip
excluded rows silently and write to it when they exclude.

### Deliverables

```
deploy/.model-status.json        new (gitignored) -- the exclusion ledger
scripts/_model_status.py         new -- ledger I/O: load, record_exclusion,
                                        is_excluded, prune_to_catalog,
                                        host_env stamping (reuse bench
                                        capture_host_env / host_env_id)
scripts/select-models.py         modify -- write fit-window exclusions to the
                                          ledger; skip ledger-excluded rows
scripts/_probe_hf_common.py      modify -- skip ledger-excluded rows (no
                                          "not on disk" line); write
                                          unsupported_arch/oom exclusions
.gitignore                       modify -- ignore deploy/.model-status.json
docs/backends.md                 modify -- document the ledger + reasons
tests/python/test_model_status.py new -- schema, keying, gating, sha-stability
```

Ledger shape (schema v1):

```
{
  "_meta": { "schema_version": 1, "host_env_id": "<12-char>",
             "host_vram_gb": 24, "updated_at": "<iso>" },
  "models": {
    "gemma-4-31b-it": {
      "repo": "google/gemma-4-31b-it", "backends": ["vllm","sglang"],
      "status": "excluded", "reason": "too_big",
      "detail": "est 58.2 GB > 24 GB budget",
      "judged_at": { "host_vram_gb": 24, "ctx": 32768 },
      "sha_stable": true, "last_sha": null, "updated_at": "<iso>"
    },
    "gemma-4-e4b-it": { "...": "reason: too_small (< 12 GB floor)" },
    "gemma-4-e2b-it": { "...": "reason: unsupported_arch, sha_stable: true" }
  }
}
```

### Detailed steps

1. `scripts/_model_status.py`: keyed by catalog `name`; `reason in
   {too_big, too_small, unsupported_arch, oom, manual}`; stamp `host_env_id`
   (reuse `scripts/bench/_bench_core.capture_host_env`) so a ledger built on
   a different GPU is auditable / ignorable. `is_excluded(name, backend,
   vram, ctx)` honors sha-stability (open question 2).
2. `select-models.py`: where the VRAM window rejects a row
   (`build_rows`, `:1232`; prune path `:1421`), call
   `record_exclusion(name, too_big|too_small, ...)`. Before considering a
   row a candidate, `if is_excluded(...) : continue`.
3. `_probe_hf_common.run_probe_pass`: before the "not on disk" skip
   (`:1560`), `if is_excluded(...)`: skip silently. When a probe yields a
   terminal verdict, `record_exclusion(...)`.
4. `prune_to_catalog`: drop ledger rows no longer in the catalog (a model
   removed from `model-families.yaml`). **Shipped in `scripts/model-sync.py:main()`**
   (not in the probers), non-dry-run only, against the unfiltered
   catalog, with two guards: an empty catalog is a no-op, and a prune
   that would remove more than half the ledger is refused.
5. Host-VRAM change invalidates the ledger: stamp `_meta.host_vram_gb`; if it
   differs from the current budget, treat the ledger as advisory (re-derive)
   rather than authoritative -- a 24 GB exclusion is wrong on an 80 GB host.

### Exit criteria

- After one `make model-pull` + `make probe-vllm`, `deploy/.model-status.json`
  lists gemma-4-31b-it (too_big), gemma-4-e4b-it (too_small), gemma-4-e2b-it
  (unsupported_arch).
- A second `make probe-vllm` prints no "not on disk" / no re-probe for those
  rows.
- The ledger never causes a fitting model to be skipped (unit-tested).

### Phase 3 risks

| Risk                                          | Mitigation                                                  |
| --------------------------------------------- | ---------------------------------------------------------- |
| Ledger and probe caches disagree              | Probe caches stay authoritative for POSITIVE fit; ledger only adds exclusions + mirrors terminal verdicts |
| Stale ledger after a GPU upgrade              | `_meta.host_vram_gb` guard re-derives when the budget changes |
| Silent over-exclusion hides a model           | `make model-status` (read-only) prints the ledger with reasons; `--clear <name>` un-excludes |

---

## Phase 4 -- model-sync (closed loop)

### Goal

`make model-sync` makes the on-disk + probed set equal to "every catalog row
that fits and loads on this host," onboarding genuinely new rows
automatically.

### Deliverables

```
scripts/model-sync.py            new -- the diff + onboard driver
Makefile                         modify -- `make model-sync` (+ DRY_RUN,
                                          SYNC_MAX_DOWNLOADS, FAMILY scope)
docs/backends.md                 modify -- document the loop
tests/python/test_model_sync.py  new -- diff logic (new vs excluded vs
                                          serving), budget cap, dry-run
```

### Detailed steps

1. `make model-sync` runs: `catalog-regen` -> load catalog + ledger + probe
   caches -> classify every row:
   - in ledger as excluded -> skip;
   - has a probe-passed cell (serving) -> skip;
   - fails the fit window (fresh math) -> record exclusion, skip;
   - else GENUINELY NEW -> queue for onboard.
2. For each queued row (up to `SYNC_MAX_DOWNLOADS`): `model-pull NAME=<row>`
   -> the relevant `probe-*` scoped by `PROBE_REPO` -> read back the verdict
   -> record serving (fit) or exclusion (oom/unsupported_arch). A transient
   `infra` is left un-recorded so the next sync retries it.
3. `DRY_RUN=1` prints the plan (new / serving / excluded counts + the
   onboard queue) without downloading.
4. GPU exclusivity: the HF probe step requires backends down; `model-sync`
   asserts it (reuse `assert_no_active_backends`) and sequences
   ollama-vs-HF phases so it does not fight the GPU with itself.

### Exit criteria

- Adding a fitting NVFP4 row to `model-families.yaml` then running
  `make model-sync` downloads + probes it and leaves it `serving`, with no
  other row touched.
- A newly added 400B row is recorded `excluded{too_big}` without downloading.
- `DRY_RUN=1` mutates nothing.

### Phase 4 risks

| Risk                                         | Mitigation                                                |
| -------------------------------------------- | --------------------------------------------------------- |
| Unattended sync fills the disk               | `SYNC_MAX_DOWNLOADS` hard cap; DRY_RUN default in docs    |
| A flaky `infra` model retried every sync     | Acceptable -- infra is genuinely transient; an operator can `manual`-exclude it via the ledger |
| Sync races a live serving backend on the GPU | `assert_no_active_backends` precondition, same as probe-* |

---

## Combined risk register

| Risk                                                   | Phase | Mitigation                                                       |
| ------------------------------------------------------ | ----- | ---------------------------------------------------------------- |
| Mis-tuned arch patterns flip transient <-> terminal    | 1     | Bidirectional unit tests; patterns are exact load-error strings  |
| Ledger drifts from reality (caches authoritative)      | 3     | Ledger is exclusions-only overlay; probe caches own positive fit |
| Host-VRAM change silently invalidates verdicts         | 3     | `_meta.host_vram_gb` guard re-derives on mismatch                |
| Closed loop downloads too much unattended              | 4     | Per-run download budget + dry-run                                |

## Migration / rollback story

- Rollback = revert the PRs. The ledger is additive and gitignored; deleting
  `deploy/.model-status.json` returns the system to today's behaviour
  (re-derive exclusions each run). The probe caches are untouched in shape.
- No existing-user migration: the ledger is born empty and populated on the
  next `model-pull` / `probe-*` / `model-sync`. Phase 1/2 cache changes are
  backward-compatible (no schema bump; only classification + carry-forward).
- Existing `make probe`, `make probe-vllm`, `make model-pull` keep working
  with identical flags; the ledger only adds skips, never changes a positive
  verdict.

## Estimated effort

| Phase    | Engineering effort                 | Wall-clock |
| -------- | ---------------------------------- | ---------- |
| Phase 1  | 1 PR, ~60 LoC + tests              | 0.5 day    |
| Phase 2  | 1 PR, ~120 LoC + tests             | 1 day      |
| Phase 3  | 1 PR, ~300 LoC (new module) + tests| 2 days     |
| Phase 4  | 1 PR, ~250 LoC (new driver) + tests| 2 days     |
| Total    | 4 PRs                              | ~1 week    |

## References

- `scripts/_probe_hf_common.py` -- `classify_failure_logs` (:747),
  `refresh_top_level_from_cells` (:1317), `ensure_entry` (:1250),
  `recovery_image` (:169), `run_probe_pass` skip + key logic (:1560, :1568,
  :1592).
- `scripts/select-models.py` -- fit window (:1232), `MIN_VRAM_FRACTION`
  (:1338, :1409), `_row_probe_rejected` (:924), `assign_cell_candidates`
  (:968), prune path (:1421).
- `scripts/bench/_bench_core.py` -- `capture_host_env` / `host_env_id` /
  `stamp_host_env`, reused for the ledger's host stamping.
- `docs/backends.md` -- probing procedure, cache hygiene, failure-mode
  taxonomy (the ledger extends this taxonomy with a persisted negative space).
- This plan was prompted by the gemma-4 probe run (2026-06-23): a
  `Gemma4ForConditionalGeneration` load failure mis-tagged `infra`, plus
  stranded `repo@sha` keys after `catalog-regen`.
