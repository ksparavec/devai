# Card-derived hints and the bench closed loop

_Derive backend hints from checkpoint metadata instead of hand-curating them, and close the download -> probe -> bench loop._

## Status

Draft. Not yet scheduled for execution.

## Dependencies

- [Plan: model-lifecycle-ledger](./model-lifecycle-ledger.md) -- Phase 5b extends
  the exclusion ledger (`scripts/_model_status.py`) that plan introduced. Its
  Phase 3 has shipped, which is all this plan needs.
- [Plan: bench-rewrite](./bench-rewrite.md) -- Phase 5 builds on the schema v3
  bench cache and the `_meta.host_env_history` stamping that plan introduced.
  Its Phases 1-5 have shipped; the deferred Phase 6 is not a prerequisite.

## Enables / Unblocks

- Onboarding a new model without hand-writing a `parsers:` block, which is the
  main per-model curation cost in `make model-sync` today.
- A bounded, resumable `make bench-sync`, making bench coverage a background
  chore rather than a manual sequence.
- A provenance model (`curated` / `card` / probed) that a later
  README-derived-engine-flags plan can reuse.

## Out of scope

- **README-derived engine flags and env vars.** Real value -- the Gemma-4 card
  publishes `VLLM_NVFP4_GEMM_BACKEND=marlin`, which is exactly the shape of a
  `deploy/recovery-flags.json` entry -- but `README.md` is untrusted internet
  text feeding container launch arguments, i.e. a prompt-injection path into
  `podman run`. It needs an allowlist design of its own. Revisit once Phase 3
  has proven the provenance model.
- **Any model-picker change.** The picker is a read-only consumer of these
  caches (all four are mounted `:ro` by `bin/devai-agent`); this work is
  entirely host-side.
- **Ollama-side derivation.** GGUF checkpoints have no HF chat-template layout,
  and Ollama handles reasoning and tool calling natively.
- **Extending a model past its as-delivered context.** Capacity claims found in
  a model card are explicitly ignored (see constraint 3).

## Open questions

1. Should `make bench-sync` default to `--sampling=harness` or `--sampling=card`?
   -- recommendation: `harness`, to keep existing rows comparable; `card` is
   opt-in per run.
2. Should a `bench_dropped` verdict at ctx N suppress benching at ctx < N?
   -- recommendation: no. Apply the verdict at the judged ctx and above only,
   since long-context quality degrades rather than improves.
3. Should Phase 1's metadata fetch be a `make` target or a one-off script?
   -- recommendation: a `make` target, so re-validating after a rule change is
   cheap and reproducible.

## Context

Two gaps surfaced while assessing whether the CLI-only `make` features could be
driven automatically.

**Checkpoint metadata is under-used.** Every downloaded HF checkpoint ships
`chat_template.jinja` (or an embedded `tokenizer_config.json` template),
`generation_config.json`, `hf_quant_config.json` and `README.md`. The probers
read `config.json` and nothing else (`scripts/_probe_hf_common.py:340-423`,
`scripts/_probe_load.py:399`). Everything a model states about its own
tool-call format, reasoning markup and sampling defaults is discarded, and is
instead hand-curated into the `parsers:` blocks of
`scripts/model-families.yaml`.

The cost is recorded in the repo itself. `scripts/model-families.yaml:144-152`
notes that the Gemma-4 tool parser had to be inferred from a sibling family,
and that until it was, "the router strips tools/tool_choice and the model
scored 0 on the tools bench despite tool-calling fine (its Ollama GGUF twin
scored tools=1.00)". That value is mechanically recoverable from the
checkpoint's own `chat_template.jinja`, which contains the tool-call markup the
model was trained to emit.

A read-only experiment over the five downloaded HF models predicted the vLLM
tool parser from template markup alone and matched the curated value 5/5
(`gemma4`, `nemotron_json`, `qwen3_xml` x2, `openai`). That result is
**in-sample** -- the discriminator rules were written after reading those five
templates -- so it must be validated out-of-sample before anything consumes it.

**There is no closed loop for bench.** `make model-sync` closes the
download -> probe loop. Bench has no equivalent, so populating the leaderboard
is a manual sequence whose precondition reverses mid-way: `probe-*` requires
the stack **stopped** (`Makefile:1211`), `bench-*` requires it **running**
(`Makefile:1622`). Nothing consumes the drop-trigger verdicts or the host-env /
image-digest stamps that already exist, so there is no way to ask "what still
needs benching, and what became stale".

## Approach

Add one small extraction module that reads what the checkpoint already says
about itself, and thread its output through the existing probe path as a
**fallback** rather than a replacement: curated `parsers:` values always win,
derived values fill gaps, and a disagreement between the two is warned about
but never acted on. The probe remains the arbiter -- derivation changes the
starting point, not the verification. Separately, add a `bench-sync` driver
that mirrors `scripts/model-sync.py`, reusing the target-discovery logic that
already exists in the bench runner and adding only the three things it lacks:
stack-state bracketing, staleness classification from stamps already recorded,
and persistence of drop verdicts.

Phases 1-4 and Phase 5 are independent tracks and can ship in either order.
Phase 3 is gated on Phase 1's out-of-sample result.

---

## Phase 1 -- Card-hint extraction and out-of-sample validation

### Goal

A read-only, tested extraction module plus a report that shows predicted vs
curated parser values, validated against families that were not used to write
the rules. Ships without changing any launch argument.

### Deliverables

```
scripts/_card_hints.py              new      -- template / generation_config extraction + rules
scripts/card-hints-report.py        new      -- read-only predicted-vs-curated report
tests/python/test_card_hints.py     new      -- rule assertions over fixtures
tests/fixtures/card-hints/          new      -- minimal templates per curated family
Makefile                            modify   -- `card-hints`, `card-hints-fetch` targets
```

### Detailed steps

1. Write `scripts/_card_hints.py`, following the file-per-concern convention of
   `_probe_core.py` / `_model_status.py`:
   - `load_chat_template(model_dir) -> (text, source)`. Prefers
     `chat_template.jinja`; falls back to `tokenizer_config.json["chat_template"]`,
     handling the list-of-named-templates shape. Returns empty on absence and
     never raises -- absent metadata must degrade to today's behaviour.
   - `predict_tool_format(template) -> (format_class, evidence)`. Ordered
     rules, first match wins, per the discriminator table below.
   - `predict_reasoning_format(template) -> (format_class, evidence)` returning
     `harmony`, `think_delimited`, `enable_thinking_only`, or `None`.
   - `FORMAT_TO_PARSER[(format_class, backend)] -> parser_name`, the static
     per-backend name table. `think_delimited` deliberately resolves to `None`
     (ambiguous -- see the reasoning note below).
   - `sampling_defaults(model_dir) -> dict` reading `temperature`, `top_p`,
     `top_k`, `eos_token_id` from `generation_config.json`. Used in Phase 4.

   Discriminators, each validated against an on-disk checkpoint:

   | Format class | Evidence in the chat template |
   | ------------ | ----------------------------- |
   | `gemma4` | inverted delimiters, `<\|tool_call>call:NAME{..}<tool_call\|>` |
   | `qwen3_xml` | `<tool_call>` wrapping `<function=` plus `<parameter=` XML |
   | `hermes` | `<tool_call>` carrying a JSON payload, no `<function=` |
   | `nemotron_json` | `<TOOLCALL>[{"name":..,"arguments":..}]` |
   | `harmony` | `<\|channel\|>commentary` or `to=functions.` |

2. Write `scripts/card-hints-report.py` and wire `make card-hints`: for every
   model on disk, print predicted vs `model-families.yaml` vs probe-cache
   value, with the evidence string that produced the prediction. Read-only.

3. Add `make card-hints-fetch`: pull **metadata only** for the
   curated-but-not-downloaded families -- `hf download <repo> --include
   "chat_template.jinja" "tokenizer_config.json" "generation_config.json"`.
   A few hundred KB, no weights, no GPU. Stage under
   `~/.cache/devai/card-hints/` (honouring `XDG_CACHE_HOME`), **not** under
   `/var/cache/devai/` -- per the mount-point convention in CLAUDE.md, a new
   top-level directory there is not volume-backed.

   Families to validate against: `deepseek-r1-distill`, `llama3.1`, `qwen3`,
   `qwen3-coder`, `nemotron-3-nano`, `diffusiongemma`.

4. Reduce the fetched templates to minimal fixtures under
   `tests/fixtures/card-hints/` and assert the predicted parser for each in
   `tests/python/test_card_hints.py`.

### Exit criteria

- `make card-hints` reproduces the 5/5 tool-parser match on the downloaded
  models, each with its evidence string.
- `make test-python` passes the new fixture cases for every curated family,
  including those not downloaded.
- `qwen3_coder` either gains a discriminator from its fetched template, or is
  documented as requiring curation. It currently has none.

### Phase 1 risks

| Risk | Mitigation |
| ---- | ---------- |
| Rules were fitted in-sample (n=5) | This phase exists to test that. A poor out-of-sample result stops Phase 3; the report stays useful as an advisory tool. |
| A family's newer members change markup | The report surfaces predicted vs curated disagreement, which is exactly this signal. |
| HF metadata fetch unavailable (air-gapped) | Fixtures are committed, so the tests run offline. Only re-validation needs network. |

---

## Phase 2 -- Gemma-4 reasoning discrepancy

### Goal

Settle a suspected second scoring miss on the model that already cost a tools
bench score, before any derivation is wired in.

### Deliverables

```
docs/backends.md                    modify   -- record the finding
scripts/model-families.yaml         modify   -- only if the finding warrants it
scripts/_probe_hf_common.py         modify   -- only if a classifier gap is confirmed
```

### Detailed steps

1. `Gemma-4-26B-A4B-it-NVFP4`'s template gates a `<|think|>` token on
   `enable_thinking` (5 occurrences plus a `strip_thinking()` macro), yet
   `model-families.yaml` omits a reasoning parser for the family and the probe
   cache records `capability=none`. Launch the model once under vLLM with
   `chat_template_kwargs.enable_thinking=true` -- the request shape
   `_probe_hf_common.py:604-651` already constructs -- and inspect whether
   `<|think|>` appears in the output.
2. If it emits: this is a probe-detection gap, not a curation gap. Fix the
   capability classifier and add a regression case.
3. If it does not emit: record that in the family block so the next reader does
   not re-investigate.

### Exit criteria

- The launch output is recorded verbatim in the finding. No claim either way
  without it.

### Phase 2 risks

| Risk | Mitigation |
| ---- | ---------- |
| Costs GPU time and stack downtime | One launch, one request. Schedule alongside any other probe run. |

---

## Phase 3 -- Wire derived parsers as fallback

### Goal

Uncurated and newly onboarded models get a probe-verified parser without a
human writing a `parsers:` block, with zero behaviour change for curated
families.

### Deliverables

```
scripts/_probe_hf_common.py         modify   -- fallback + provenance at 1758-1760
docs/backends.md                    modify   -- derivation and provenance fields
CLAUDE.md                           modify   -- new script + target in the key-files map
```

### Detailed steps

1. `scripts/_probe_hf_common.py:1758-1760` inside `run_probe_pass()` is the
   single site where curated parsers are resolved from the catalog row, and
   both `name` and `models_dir` are already in scope. Change the resolution to:

   ```
   curated = (row.get("parsers") or {}).get(spec.name, {}).get("tool") or None
   derived = card_hints.derive_parser(name, models_dir, spec.name, "tool")

   if curated is None:
       tool_parser, tool_source = derived, ("card" if derived else None)
   else:
       tool_parser, tool_source = curated, "curated"
       if derived and derived != curated:
           warn(...)          # disagreement; behaviour unchanged
   ```

   Same shape for the reasoning parser, which by design will usually derive
   `None`.

2. Record provenance on the probe-cache entry: `tool_parser_source` and
   `reasoning_parser_source` in `("curated", "card", None)`, plus
   `parser_disagreement` when the two differ. Additive fields only -- no schema
   bump, consistent with how the LOAD probe augments cells.

3. Extend the existing `parser_label` log line in `run_probe_pass` to show the
   source, so probe logs stay self-explaining.

4. Surface disagreements in `make card-hints` output so curation drift is
   visible rather than silent.

### Exit criteria

- `make probe-vllm PROBE_REPO=<curated model> PROBE_FORCE=1` leaves the parser
  value unchanged and records `tool_parser_source=curated`.
- The same on a model with no curated parser records `tool_parser_source=card`,
  and the probe's own tool round-trip confirms the derived value.
- A no-op re-probe diff shows only the new provenance fields.
- `make test-probe-ollama-idempotent` remains byte-identical (the Ollama path
  is untouched).

### Phase 3 risks

| Risk | Mitigation |
| ---- | ---------- |
| A rule bug degrades a working model | Curated values always win; derivation only fills gaps. |
| Unknown markup yields no parser | That is exactly today's behaviour (no parser flags emitted). Degrades safely. |
| Reasoning parser names stay ambiguous | By design: `think_delimited` resolves to `None` rather than guessing between `qwen3` / `nemotron_v3` / `deepseek_r1` / `nano_v3`. |

---

## Phase 4 -- Bench sampling parity

### Goal

Make the sampling policy used by the bench harness explicit and recorded,
instead of accidental.

### Deliverables

```
scripts/_probe_hf_common.py         modify   -- stamp `card_sampling` into the probe cache
scripts/bench/bench_runner.py       modify   -- `--sampling={harness,card}`, record mode on the row
scripts/bench/bench_report.py       modify   -- surface the mode
Makefile                            modify   -- `BENCH_SAMPLING`
docs/bench-results.md               modify   -- record the policy decision
```

### Detailed steps

1. No sampling parameters are set anywhere in `bench_runner.py` today (only
   `max_output_tokens` for longctx, line 729), so every model is benched at
   inspect_ai defaults while its card may specify otherwise -- Gemma-4 ships
   `temperature=1.0, top_k=64, top_p=0.95`. This is a bias in numbers already
   published.

2. The bench container cannot read the checkpoint: `BENCH_CACHE_MOUNTS`
   (`Makefile:1593-1596`) mounts `scripts/` ro, `deploy/` rw and the bench dir,
   but **not** the model weights. Do not add a weights mount. Instead:
   - the prober (host-side, `Makefile:1224`, already has `--models-dir`) stamps
     `card_sampling: {temperature, top_p, top_k}` from
     `_card_hints.sampling_defaults()` into the probe-cache entry;
   - the bench runner reads it from `/deploy`, which it already mounts.

3. Add `--sampling={harness,card}` to `bench_runner`, defaulting to `harness`
   so existing rows stay comparable, plus `BENCH_SAMPLING` in the Makefile.
   Record the mode actually used on every row so mixed-mode caches remain
   auditable.

### Exit criteria

- `make bench-vllm BENCH_REPO=<one model> BENCH_TASKS=gsm8k` run at each mode
  produces two rows, each recording its own mode.
- `make bench-report` shows the mode, and the policy decision is written down
  in `docs/bench-results.md`.

### Phase 4 risks

| Risk | Mitigation |
| ---- | ---------- |
| Existing rows become incomparable | Default stays `harness`; `card` is opt-in and labelled per row. |
| Card defaults are absent for some checkpoints | `sampling_defaults()` returns an empty dict; the runner falls back to harness defaults. |

---

## Phase 5 -- bench-sync closed loop

**Status: Shipped 2026-07-26.** `scripts/bench-sync.py` + `make bench-plan`
/ `make bench-sync`, 29 tests in `tests/python/test_bench_sync.py`,
documented in docs/backends.md "The bench closed loop".

One correction to the design below. Step 2 reasoned that leaving
`is_excluded()`'s reason allowlist untouched would make the new bench
reasons fail open "by construction". That was true of the hand-written
literal the plan was drafted against, but the allowlist is now DERIVED
from `VALID_REASONS` (the fix for `retired` silently failing open), so
merely adding a reason opts it INTO gating -- the exact opposite. The
implementation subtracts `_BENCH_REASONS` from the derived tuple
explicitly, and a test pins it.

### Goal

`make bench-sync` populates and refreshes the leaderboard as one bounded,
resumable, idempotent host job.

### Deliverables

```
scripts/bench-sync.py               new      -- plan_bench() + execute()
scripts/_model_status.py            modify   -- bench verdict classes + is_bench_excluded()
scripts/bench/bench_runner.py       modify   -- stamp backend image digest on each row
Makefile                            modify   -- `bench-plan`, `bench-sync`, `BENCH_MAX_TARGETS`
tests/python/test_bench_sync.py     new      -- classification tests
docs/backends.md                    modify   -- the loop and its preconditions
```

### Detailed steps

1. **`bench-plan` first (read-only).** `scripts/bench-sync.py` with
   `plan_bench()` mirroring `scripts/model-sync.py:plan_sync()`. Reuse
   `bench_runner.discover_models()` (line 204) for the target set -- it already
   diffs the probe cache and honours `serving_ok is not False`. Do not rebuild
   the diff engine. Classify each target as:
   - **new** -- fitting cell, no bench row
   - **incomplete** -- row exists, tasks missing
   - **stale(env)** -- row `host_env_id` != `_meta.current_host_env_id`
   - **stale(image)** -- backend image digest moved since the row was written
   - **dropped** -- row carries a `drop_flag`
   - **excluded** -- ledger says so

   Prerequisite for `stale(image)`: bench rows must carry the backend image
   digest. Stamp it at write time alongside the existing `host_env_id`, and
   mirror the comparison in `scripts/probe-check.py`. Rows without it classify
   as `unknown`, never as stale.

2. **Ledger extension.** In `scripts/_model_status.py`:
   - add reasons `bench_dropped` and `bench_failed`. `record_exclusion()`
     already accepts a `ctx=` kwarg, stored under `judged_at.ctx`.
   - add a **separate** query `is_bench_excluded(ledger, name, backend, *, ctx,
     sha)`. Bench verdicts must not gate download or probe -- a model dropped
     for a leak is still downloadable and still probeable. Leaving
     `is_excluded()`'s reason allowlist untouched means the new reasons fail
     open there by construction (`_model_status.py:126-127`).
   - stability: `bench_dropped` is a quality verdict -- VRAM-independent,
     sha-dependent (a re-quant re-checks). It applies at the judged ctx and
     above.
   - **preserve the existing invariant.** `bench_runner.py:515-517` states that
     the drop flag "never deletes weights or edits the exclusion ledger -- that
     stays an explicit operator action". `bench-sync` therefore writes ledger
     entries only under an explicit `--record-drops`; the default reports the
     drop and leaves the ledger alone.

3. **Orchestration.** `make bench-sync` sequences the state transition that is
   done by hand today:

   ```
   [optional catalog-regen] -> model-pull -> cache-down
     -> probe-vllm / probe-sglang / probe-load-* / probe
     -> cache-up
     -> bench-vllm / bench-sglang / bench-ollama
     -> bench-report
   ```

   - `DRY_RUN=1` runs `bench-plan` only, mirroring `model-sync`.
   - `BENCH_MAX_TARGETS` plus a wall-clock budget, mirroring
     `SYNC_MAX_DOWNLOADS`.
   - Compose from the existing filters `BENCH_REPO` / `BENCH_CTX` /
     `BENCH_TASKS` (`Makefile:1601-1606`) rather than adding new ones.
   - Group by precondition -- all probes, then one `cache-up`, then all benches
     -- so a mixed queue never thrashes the stack.
   - The target must announce that it is a long-running exclusive job: it stops
     the stack, so it cannot overlap an interactive session.

### Exit criteria

- `make bench-plan` classifies every row in the current cache, matching a
  hand-audit of `deploy/.bench-cache.json`.
- `make bench-sync DRY_RUN=1` performs no container action, verified with
  `make cache-status` before and after.
- `make bench-sync BENCH_MAX_TARGETS=1` on a clean cache completes one target
  end-to-end and leaves the stack **up**.
- Interrupting mid-run and re-running resumes without redoing completed tasks.
- The ledger is untouched without `--record-drops`; with it, `make model-status`
  shows the verdict and `CLEAR=` removes it.
- `make test-devai-tools` passes -- `devai-tools/internal/modelcache` reads
  these caches and must tolerate the new additive fields.

### Phase 5 risks

| Risk | Mitigation |
| ---- | ---------- |
| Long unattended job leaves the stack down | Bracketing always ends with `cache-up`; exit criteria test this explicitly. |
| Auto-excluding a model on a transient failure | `bench_failed` retries once; only `bench_dropped` (a quality verdict) persists, and only under `--record-drops`. |
| Drop verdict over-applied across contexts | Verdict applies at the judged ctx and above only (open question 2). |

---

## Combined risk register

| Risk | Phase | Mitigation |
| ---- | ----- | ---------- |
| Derivation rules fitted in-sample | 1 | Out-of-sample gate; Phase 3 does not ship if it fails. |
| Silent regression of a curated family | 3 | Curated always wins; disagreement warns but never acts. |
| Cache readers break on new fields | 3, 4, 5 | All new fields additive; `make test-devai-tools` covers the Go readers. |
| Bench orchestration disrupts an interactive session | 5 | Target refuses to run unattended without an explicit flag. |

## Migration / rollback story

- Phases 1 and 2 add no runtime behaviour. Rollback is reverting the PR.
- Phase 3 is behaviourally inert for every model that has a curated parser
  today, which is every model currently in the catalog. Rollback is reverting
  the resolution block at `_probe_hf_common.py:1758-1760`; the added
  provenance fields are ignored by every existing reader.
- Phase 4 defaults to today's behaviour (`harness`). Rows benched under `card`
  are labelled, so a revert leaves an auditable trail rather than silent
  inconsistency.
- Phase 5 adds new targets only; existing `bench-*` targets are unchanged. The
  new ledger reasons fail open in `is_excluded()`, so a partial revert cannot
  block downloads or probes.
- No cache migration is required at any phase -- all schema changes are
  additive.

## Estimated effort

Ballpark, not measured.

| Phase   | Engineering effort | Wall-clock |
| ------- | ------------------ | ---------- |
| Phase 1 | 1 PR, ~400 LoC incl. tests | 1-2 days |
| Phase 2 | investigation, ~0-80 LoC | half a day plus one GPU slot |
| Phase 3 | 1 PR, ~120 LoC | 1 day |
| Phase 4 | 1 PR, ~150 LoC | 1 day |
| Phase 5 | 2 PRs, ~500 LoC incl. tests | 3-4 days |
| Total   | 5-6 PRs | ~1.5 weeks |

## References

- `scripts/model-families.yaml:144-152` -- the Gemma-4 tools-bench miss that
  motivates Phase 1.
- `scripts/_probe_hf_common.py:1758-1760` -- the single parser resolution site.
- `scripts/bench/bench_runner.py:204` -- `discover_models()`, the existing diff
  engine reused by Phase 5.
- `scripts/bench/bench_runner.py:433, 515-517` -- the drop trigger and the
  operator-action invariant Phase 5b preserves.
- `Makefile:1211, 1622` -- the opposing stack preconditions for probe and bench.
- `Makefile:1593-1596` -- `BENCH_CACHE_MOUNTS`, the constraint behind the
  Phase 4 split.
- `scripts/_model_status.py:126-127` -- the fail-open reason allowlist that
  keeps bench verdicts out of download and probe eligibility.
- CLAUDE.md, "`/var/cache/devai/` mount-point convention" -- why Phase 1 stages
  fetched metadata under `~/.cache/devai/`.
