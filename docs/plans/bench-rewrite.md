# Bench Harness Rewrite -- Per-Context Rows

## Status

**Approved.** All five design decisions confirmed -- see
"Confirmed decisions" below. Ready for implementation in a single
working session (~3-4 hours of code + ~45-60 min re-bench sweep,
both required).

**Amended 2026-05-14**: Phase 6 (backfill) promoted from optional
to required. Rationale captured in the "Phased rollout" section.

## Dependencies

None.

## Enables / Unblocks

- Apples-to-apples comparison of `tok/s` across models that fit
  different context tiers. (We just discovered gpt-oss-20b's
  139.2 tok/s row was measured at ctx=262144 while
  Nemotron-3-Nano-30B-A3B-NVFP4's 143.8 tok/s row was at
  ctx=131072 -- they were never comparable, but the leaderboard
  treated them as such.)
- Context-tagged leaderboard in `docs/bench-results.md`.
- Picker shows bench numbers matching the ctx the user is about to
  launch (not a 256K number for a 32K request). With MTP now in
  play, the same model at 32K vs 128K can differ by 18 tok/s on
  decode -- the picker must not silently substitute.
- Foundation for a future "speedup-vs-context" curve that the MTP
  doc currently sketches as predicted rather than measured.

## Out of scope

- Profiling the bench harness itself (timing the harness).
- Cross-host comparison (already handled by per-row `host_env_id`
  stamping -- v3 inherits unchanged).
- Re-running historical benches against a different vLLM image
  (separate exercise; tracked alongside the MTP/Gemma4 image
  bumps).
- New bench tasks beyond `gsm8k_subset_100`, `humaneval_subset_50`,
  `tools_use_20`, `leak_probe`, `longctx_probe`.
- Acceptance-rate measurement under MTP (separate "MTP probe
  detail" plan; the MTP feature work just landed records only
  `mtp_overhead_gb` and `mtp_fits`).
- Picker UI changes beyond the bench-data lookup path. Existing
  columns (`TPS`, `CODE%`, `REAS%`, `TOTAL%`, `LEAK%`) stay; they
  just become accurate at the user's chosen ctx.
- Renaming or relocating `deploy/.bench-cache.json`.

## Confirmed decisions

All five points below were confirmed on 2026-05-14 before
implementation began. Future deviations require an explicit plan
amendment.

1. **Row keying: per-ctx rows.** Cache key format
   `<base>::<backend>::<ctx>`. Each ctx tier is its own row with
   its own `metrics` and `tasks` blocks. Task-score duplication
   across ctx rows is accepted (~700 bytes per row; tens of KiB
   total at full coverage). Rationale: simplest consumer code on
   both the picker and report sides, and the longctx_probe task
   is intrinsically ctx-specific anyway so the row-per-ctx model
   matches the data shape.
2. **Runner default: pick largest fitting ctx; opt-in `--ctx` to
   scope.** Identical to today's behavior so `make bench` wall
   time stays at one cell per model. `--ctx 32K,128K` or
   `--ctx 32K` scopes to specific tiers; `--all-ctx` iterates
   every fits=true cell in the probe cache.
3. **Migration: in-place stamp v2 rows with recovered ctx from
   the 2026-05-05 router log.** The nine existing v2 rows all
   have unambiguous evidence (table in the Context section
   below). Migration is hardcoded in `RECOVERED_CTX_MAP` inside
   `_bench_core.py`, runs at first writer invocation,
   idempotent. Preserves history at zero re-bench cost.
4. **Picker fallback: show `-` when no bench row exists at the
   user's current ctx.** Picker preview pane adds one line
   (`Bench: not available at ctx=<N> (run \`make bench --ctx <N>\`)`)
   when the lookup misses. No nearest-ctx substitution, no
   asterisks. Honest about the gap; prompts the operator to fill
   it.
5. **Schema strictness: graceful read of pre-v3 rows.** Reader
   logs a one-line warning on encountering a row without a
   `context` field or with `context=0`, then treats it as "no
   data at this ctx" (returns the empty marker in the picker).
   Mirrors the probe-cache loader's v1-tolerance pattern at
   `gpu-arbiter/main.go:357`. Operator can re-bench at
   convenience.

## Context

The bench-cache today (`deploy/.bench-cache.json`, schema v2)
records `tps_sustained_p50`, `ttft_ms_*`, `mean_vram_gb`,
`peak_vram_gb`, `vllm_kv_cache_usage_perc`, and task scores per
`<repo>@<sha>::<backend>` row. **It does not record the context at
which any of those were measured.** The bench runner picks the
largest fitting context from the probe cache and sends a
`<name>@<ctx>` override to the router; the router writes one
container at that ctx; the bench measurements run against it; the
ctx evaporates by the time the row hits the cache file.

The MTP work that just landed turns this from a theoretical bug
into a load-bearing one:

- `Qwen3.5-9B-NVFP4` baseline: 57.1 tok/s at 32K, 56.9 tok/s at
  128K -- 0.4 tok/s gap, fine to gloss over.
- `Qwen3.5-9B-NVFP4` + MTP: 104.3 tok/s at 32K, 86.8 tok/s at
  128K -- **17.5 tok/s gap**, cannot be glossed over.

The picker today would show "104 tok/s" against this row even when
the user is about to launch at 128K, off by 17 tok/s. After MTP,
"the bench number depends on ctx" is a first-class concern.

We have hard evidence of the silent ctx drift in the existing 9
rows. Grep of `/var/cache/devai/logs/devai-router.log` for the May
5 bench window produces the launch ctxs verbatim:

| Cache row | Launched at |
|---|---|
| `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`        | **32 K**  |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`         | **64 K**  |
| `nvidia/Llama-3.1-8B-Instruct-NVFP4`              | **128 K** |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`     | **128 K** |
| `nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4`         | **64 K**  |
| `nvidia/Qwen3-14B-NVFP4`                          | **64 K**  |
| `nvidia/Qwen3-8B-NVFP4`                           | **128 K** |
| `openai/gpt-oss-20b`                              | **256 K** |
| `ykarout/Qwen3.5-9B-NVFP4`                        | **128 K** |

Three different ctx tiers (32 K, 64 K, 128 K, 256 K) silently
mixed into one leaderboard. The 256 K outlier (`gpt-oss-20b`) was
specifically the one whose tok/s I claimed earlier in the MTP
session was "at 128 K" -- and I was wrong.

## Design

### 1. Cache row schema (v3)

Per-ctx rows. Key format:

    <base>::<backend>::<ctx>

where `<base>` is the existing `<repo>@<sha>` (HF) or `<digest>`
(Ollama) prefix, and `<ctx>` is the integer max-model-len the
container was launched at.

Example v3 row (after migration):

```json
"openai/gpt-oss-20b@6cee5e81ee83::vllm::262144": {
  "schema_version": 3,
  "model": "gpt-oss-20b",
  "backend": "vllm",
  "context": 262144,
  "host_env_id": "ea4fd7e7b668",
  "router_endpoint": "http://devai-router:11435",
  "first_benched_at": "2026-05-05T16:46:36+00:00",
  "last_benched_at":  "2026-05-05T16:48:43+00:00",
  "metrics": {
    "tps_sustained_p50": 139.2,
    "ttft_ms_first":     53918.3,
    "ttft_ms_steady_p50": 50.3,
    "ttft_ms_steady_p95": 54.0,
    "peak_vram_gb":      22.37,
    "mean_vram_gb":      19.97,
    "n_latency_samples": 40,
    "vllm_kv_cache_usage_perc": 0.0,
    "vllm_num_preemptions_total": 0.0,
    "vram_samples":      220
  },
  "tasks": {
    "gsm8k_subset_100":  { ... },
    "humaneval_subset_50": { ... },
    "tools_use_20":      { ... },
    "leak_probe":        { ... }
  }
}
```

Two new fields vs v2: `context` (int, mandatory in v3) and a `::ctx`
suffix on the row key.

### 2. Runner behaviour

The discovery layer (`bench_runner.py:_has_fitting_cell` -> rename
to `_fitting_ctxs`) returns a sorted list of all probe-cache cells
where `fits=true` at the host VRAM band.

`discover_models` takes a new `ctx_filter: list[int] | None` arg:

- `None` (default): one target per (model, largest fitting ctx).
  Behaviour identical to today.
- explicit list (e.g. `[32768, 131072]`): one target per
  (model, ctx) pair in the intersection of fitting and requested.
  Missing-cell pairs are skipped with a stderr note.
- `--all-ctx` boolean: shorthand for "iterate all fitting cells";
  one target per (model, ctx) for every fits=true cell.

CLI argparser additions:

```
--ctx <list>       comma-separated ctx tiers (e.g. "32K,128K"). Default: largest fitting.
--all-ctx          shorthand for iterating every fits=true ctx per model.
```

`bench_one_target` already prints `(backend=vllm, ctx=N)`; that line
becomes the leading identifier in the per-row stdout.

### 3. Migration

`scripts/bench/_bench_core.py:migrate_bench_cache_keys` already
handles v1 -> v2 (key rename to include `::<backend>` suffix).
Extend with v2 -> v3:

For each existing v2 row whose key is `<base>::<backend>` without a
trailing `::<ctx>`:

1. Look up the row's `model` field in `RECOVERED_CTX_MAP` (defined
   below).
2. If found: rewrite the key to `<base>::<backend>::<ctx>` and add
   `"context": <ctx>` at the row level. Bump `schema_version` to 3.
3. If not found: leave the row in place with `"context": 0,
   "schema_version": 3, "_migration_warning": "ctx not recovered;
   re-bench to populate"`. Reader treats ctx=0 as
   "pre-migration / unknown".

`RECOVERED_CTX_MAP` (hardcoded in `_bench_core.py`, never edited
after this plan ships -- it captures historical fact, not policy):

```python
# 2026-05-05 router log evidence; never grows after migration.
RECOVERED_CTX_MAP: dict[str, int] = {
    "DeepSeek-R1-Distill-Llama-8B":            32768,
    "DeepSeek-R1-Distill-Qwen-7B":             65536,
    "Llama-3.1-8B-Instruct-NVFP4":            131072,
    "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4":   131072,
    "NVIDIA-Nemotron-Nano-9B-v2-NVFP4":        65536,
    "Qwen3-14B-NVFP4":                         65536,
    "Qwen3-8B-NVFP4":                         131072,
    "Qwen3.5-9B-NVFP4":                       131072,
    "gpt-oss-20b":                            262144,
}
```

Migration runs at first writer invocation after the upgrade.
Idempotent (re-running it on already-v3 rows is a no-op).

### 4. Picker consumer

`scripts/model-picker.py:_load_bench_records` changes return type:

```python
# v2 today
def _load_bench_records(paths) -> dict[tuple[str, str], dict]: ...

# v3 after
def _load_bench_records(paths) -> dict[tuple[str, str, int], dict]: ...
```

Lookup site at `model-picker.py:2075` becomes:

```python
ctx = int(m.get("_picker_context") or 0)
bench_key = (model_name, backend, ctx)
bench_row = bench_records.get(bench_key)
```

When `bench_row is None`, the `TPS` / `CODE%` / `REAS%` / `TOTAL%`
/ `LEAK%` columns render `-` in the row. The preview pane gains
one line:

    Bench: not available at ctx=<N> (run `make bench --ctx <N>`)

Otherwise (bench row exists), columns populate from the matched
row -- which is now guaranteed to be at the user's ctx.

### 5. Report rendering

`scripts/bench/bench_report.py` adds a `CTX` column to the
leaderboard table. Models benched at multiple ctx tiers produce
multiple table rows, indented or grouped (sketch below):

```
MODEL                          CTX    TPS      TTFT_p50   PEAK_VRAM   ...
gpt-oss-20b                    32K    -        -          -
gpt-oss-20b                    64K    -        -          -
gpt-oss-20b                    128K   -        -          -
gpt-oss-20b                    256K   139.2    50.3 ms    22.37 GB
Nemotron-3-Nano-30B-A3B-NVFP4  128K   143.8    46.9 ms    22.40 GB
Qwen3.5-9B-NVFP4               32K    -        -          -
Qwen3.5-9B-NVFP4               128K   55.27    31.5 ms    21.50 GB
```

`-` cells are an explicit "not benched at this ctx" signal -- they
prompt the operator to run a targeted sweep. Models with only one
benched ctx render as today, plus the new CTX column.

## Per-file change list

### `scripts/bench/_bench_core.py` (~566 lines, est +120 / -10)

| Function / region | Change |
|---|---|
| `BENCH_CACHE_SCHEMA_VERSION` (line ~231) | Bump 2 -> 3. |
| `is_row_key(key)` (~234) | Accept new `<base>::<backend>::<ctx>` shape (split on `::`, expect 3+ segments). |
| `cache_key_for_entry(entry, backend, ctx)` (~473) | New `ctx: int` arg; returns `f"{base}::{backend}::{ctx}"`. |
| `migrate_bench_cache_keys(cache)` (~510) | Extend with v2 -> v3 logic: map each ctx-less key to its v3 form via `RECOVERED_CTX_MAP`. Idempotent. |
| New constant `RECOVERED_CTX_MAP` (top of file) | 9-entry dict; see above. |
| `_make_row(model, backend, host_env_id, ctx)` (new helper or extension of existing row builder) | Stamp `context` at row top level. |

### `scripts/bench/bench_runner.py` (~738 lines, est +60 / -15)

| Function / region | Change |
|---|---|
| `_has_fitting_cell(entry, host_vram_gb)` (~173) | Replace with `_fitting_ctxs(entry, host_vram_gb) -> list[int]` returning sorted fits=true ctxs. |
| `discover_models(backend, *, host_vram_gb, repo_filter, ctx_filter)` (~197) | New `ctx_filter` kwarg. Targets become `(model, ctx)` pairs. |
| `bench_one_target(target, ...)` (~343) | `target["ctx"]` already used; writes `context: target["ctx"]` into the row. |
| `_latency_metrics_into_row(...)` (~554) | Adds `"context": ctx` to the row top-level. |
| `migrate_bench_cache_keys` call site (~696) | Already exists; just runs the extended migrator. |
| argparser (~615 area) | New `--ctx` and `--all-ctx` flags. |

### `scripts/bench/bench_report.py` (~183 lines, est +40 / -5)

| Region | Change |
|---|---|
| Header row | Insert `CTX` column. |
| Per-row rendering | Read `row["context"]`; format as `32K` / `64K` / `128K` / `256K` notation when divisible; otherwise raw int. |
| Sort / grouping | Sort by (model, ctx) so multi-ctx benches cluster together. |
| Footer note | One paragraph: schema v3, rows ordered by `(model, ctx)`, `-` cells mean "not benched at this ctx". |

### `scripts/model-picker.py` (~2530 lines, est +30 / -10)

| Region | Change |
|---|---|
| `_BENCH_CACHE_PATHS` (~71) | Unchanged. |
| `_load_bench_records(paths)` (~280) | Return type and key tuple grow by one int. |
| Type annotations `dict[tuple[str, str], dict]` at lines 2041, 2089 | Updated to include int. |
| Bench-row lookup (~2075) | Use `(model_name, backend, picker_context)` tuple. |
| Preview pane (`_capability_summary_text`, ~1726) | When `bench_row is None`, add `Bench: not available at ctx=<N>` line. |

### `deploy/.bench-cache.json` (data file, gitignored)

Migration runs on first `bench_runner.py` invocation post-upgrade.
Idempotent. No diff (file is gitignored).

### `docs/bench-results.md` (regenerated)

Re-emit after Phase 4. Will show `-` cells for everything except
the 9 historical rows until Phase 6 (re-bench) lands.

### `CLAUDE.md` (project-level instructions)

The block at line ~340-360 describing `.bench-cache.json` schema
needs:

- Mention schema v3: row keys now include `::<ctx>`; `context`
  field at row level.
- Add a one-line note about `RECOVERED_CTX_MAP` and its purpose.
- Update the example row.

## Phased rollout

Six small commits, each green at HEAD. **All six phases (including
Phase 6 backfill) are required for the plan to be Done.** Phase 6
was originally drafted as optional; promoted to required on
2026-05-14 because the load-bearing motivation in the Context
section -- "the picker today would show 104 tok/s ... off by 17
tok/s" -- is only resolved once each historical model has bench
rows at the ctx tiers the picker actually offers, not just at
one tier each. Without backfill, the picker continues to render
`-` cells for the 9 historical models at every ctx they were
*not* originally benched at, which is the visible failure mode
this plan exists to fix.

| # | Phase | Files | Stays green by |
|---|---|---|---|
| 1 | **Schema + migrator** | `_bench_core.py`, new unit test fixture | New constants + helpers + migration function. Reader of v3 works on either v2 or v3 input (graceful). Existing tests pass. |
| 2 | **Runner** | `bench_runner.py` | New CLI flag, default preserves today's behaviour (pick largest fitting ctx). Migration runs on first write. |
| 3 | **Picker consumer** | `model-picker.py` | Tuple key grows by one int. Pre-migration rows (ctx=0) treated as "no bench at this ctx" by lookup. |
| 4 | **Report renderer** | `bench_report.py`, regenerate `docs/bench-results.md` | New CTX column; multi-ctx benches grouped by model. |
| 5 | **Project docs** | `CLAUDE.md` | Schema description aligned with code. |
| 6 (**required**) | **Backfill bench coverage** | bench-cache writes | Run `make bench --ctx 32K,64K,128K` over the 9 historical models to populate empty cells. ~45-60 min wall time. Plan is not Done until this lands. |

Suggested commit messages (matching project style):

```
feat(bench): per-context row keying + migration (schema v2 -> v3)
feat(bench-runner): --ctx / --all-ctx flags; one target per (model, ctx)
feat(picker): bench-record lookup keyed by (model, backend, ctx)
feat(bench-report): CTX column in the leaderboard; grouped by model
docs(claude): bench-cache schema v3 description
chore(bench): backfill 32K bench data for the 9 historical models
```

## Verification

Per-phase smoke tests:

- **Phase 1**: `python3 -m py_compile scripts/bench/_bench_core.py`.
  Add a pytest-style unit test (or inline `python3 -c`) that
  builds a synthetic v2-shaped cache fixture, runs
  `migrate_bench_cache_keys`, asserts v3 keys + `context` field +
  schema bumped. Idempotency check (second run is a no-op).
- **Phase 2**: `python3 -m py_compile scripts/bench/bench_runner.py`.
  Run `make bench --ctx 32K` against one already-probed model
  (e.g., `Qwen3.5-9B-NVFP4`) and confirm exactly one row lands
  with `context: 32768` in the cache.
- **Phase 3**: open `model-picker --agent claude`, navigate to a
  model with a bench row at the picker's chosen ctx -> `TPS`
  column populates. Navigate to a model whose only bench row is
  at a different ctx -> `TPS` column shows `-` and preview pane
  shows the "not available" line.
- **Phase 4**: regenerate `docs/bench-results.md`. Confirm CTX
  column present, `-` cells in the right places, multi-ctx
  benches grouped per model.
- **Phase 5**: grep CLAUDE.md for the new schema language.
- **Phase 6**: rerun bench harness over 32K cells; confirm 9 new
  v3 rows land alongside the migrated 128K/256K ones; report
  shows both side-by-side per model.

End-to-end smoke after Phase 5 lands:

```bash
# Confirm picker shows accurate bench data at the user's ctx
DEVAI_MTP_PREVIEW=0 devai-agent --model Qwen3.5-9B-NVFP4@32768 --show
# Picker preview pane for Qwen3.5-9B-NVFP4@32K should report
# either the 32K bench (after Phase 6) or "no bench at ctx=32768"
# (before Phase 6).
```

## Risks and rollback

| Risk | Detection | Mitigation | Rollback |
|---|---|---|---|
| Migration mismatches ctx for a row | Diff migrated row's `context` against the recovered router-log value | `RECOVERED_CTX_MAP` is hardcoded, reviewable, only 9 entries | Revert migration commit; re-run after fix |
| Pre-migration v2 row not in `RECOVERED_CTX_MAP` (e.g. someone benched a 10th model that escaped my survey) | Migration warns + sets ctx=0 | Operator can re-bench at convenience; ctx=0 reader treats as "no data at this ctx" so no false bench numbers leak | Re-bench that one model |
| Picker keyed by exact `(model, backend, ctx)` misses rows benched at slightly different ctx | Visual inspection during Phase 3 verification | Document that picker matches exact ctx; users opt-in to extra ctx coverage via `make bench --ctx <N>` | Revert picker commit |
| Bench-cache file size grows due to task-data duplication across ctx rows | `du -h deploy/.bench-cache.json` | Task data is small (~700 bytes per row); 9 models × 4 ctx tiers = 36 rows ~= 25 KiB total | None needed -- minor cost vs the clarity gain |
| `--all-ctx` explodes probe wall time | `time make bench --all-ctx` | Default stays as pick-largest; explicit opt-in required; documented | None -- behaviour is opt-in |
| New `--ctx` flag fights with `--repo` or other existing filters | Argparser unit test, manual run | Specify intersection semantics: `--ctx` AND `--repo` AND `--vram` all apply | None -- it's a small param surface |
| Picker preview pane gets cluttered with "not available" lines for unbenched models | Visual review | Only render the line when the model has at least one bench row at any ctx; suppress for fully-unbenched models | Revert picker commit |

## Critical files for implementation

- `/home/sparavec/git/devai/scripts/bench/_bench_core.py`
- `/home/sparavec/git/devai/scripts/bench/bench_runner.py`
- `/home/sparavec/git/devai/scripts/bench/bench_report.py`
- `/home/sparavec/git/devai/scripts/model-picker.py`
- `/home/sparavec/git/devai/docs/bench-results.md` (regenerated; not hand-edited)
- `/home/sparavec/git/devai/CLAUDE.md` (small block update)
- `/home/sparavec/git/devai/deploy/.bench-cache.json` (auto-migrated on first writer; gitignored)

## Existing utilities to reuse

- `_bench_core.py:cache_key_for_entry` -- extend signature, preserve
  the per-backend split logic for HF vs Ollama prefixes.
- `_bench_core.py:migrate_bench_cache_keys` -- existing v1 -> v2
  migration stays as-is; v2 -> v3 logic chains after it.
- `_bench_core.py:is_row_key` -- update for the new key shape; still
  filters `_meta` out of `for k in cache` iterators.
- `_bench_core.py:save_cache` / `load_cache` -- POSIX-atomic write +
  schema version check stay unchanged.
- `_bench_core.py:capture_host_env` / `stamp_host_env` -- per-row
  `host_env_id` stamping continues unchanged for v3 rows.
- `bench_runner.py:_has_fitting_cell` -- rename + return-type
  change to `_fitting_ctxs(entry, host_vram_gb) -> list[int]`.

## Effort estimate

| Phase | Est lines | Est wall time |
|---|---|---|
| 1: schema + migrator + unit test | +120 | 1 hour |
| 2: runner + CLI flags | +60 | 30 min |
| 3: picker consumer | +30 | 30 min |
| 4: report renderer + regen | +40 | 30 min |
| 5: CLAUDE.md update | +10 | 15 min |
| Phase 6 (**required**): backfill | (cache writes only) | 45-60 min wall |
| **Total code work** | **~260 LOC + 1 doc regen** | **~3 hours code + ~1 hour bench (all required)** |

## Notes for the implementer

- The migration `RECOVERED_CTX_MAP` is a one-shot artifact. Once
  committed, no one should ever edit it -- if a new historical
  row appears later that we want to migrate, it gets ctx=0 and
  re-bench is the right answer.
- The `_meta.schema_version` field on the cache top level updates
  on first writer invocation. Reader tolerates both v2 and v3
  rows on the same load.
- `_picker_context` in the picker comes from the picker's
  context-picker step. Already populated in current code; this
  plan reuses it.
- Phase 6 re-bench is **required for the plan to be Done**
  (see Phased rollout). It still lands as its own commit so the
  schema-change commit stays small and reviewable, but the plan
  is not closed until the backfill commit also lands. Operator
  should expect a single ~4-hour working session: ~3 hours of
  code commits followed by a ~45-60 min unattended bench sweep.
