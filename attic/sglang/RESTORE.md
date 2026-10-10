# SGLang backend -- restore notes

Frozen on 2026-10-10 by platform-rebuild Phase 15, decision M16
(docs/plans/platform-rebuild.md), confirmed by the operator the same day.
I16's keep thresholds were fixed before the measurement; SGLang met three of
the four and missed the one that matters for agent sessions:

| I16 condition | measured on Qwen3.5-9B-NVFP4 | needed |
|---|---|---|
| (a) steady TTFT, SGLang / vLLM | 1.27 (45.2 / 35.7 ms) | at most 3.0 |
| (b) turn-2+ TTFT, SGLang / vLLM | **0.806** (272.8 / 338.4 ms; 95% CI 0.802-0.812) | **at most 0.75, CI below 1** |
| (c) turn-2+ TTFT, radix on / off | 0.322 (CI 0.321-0.325) | CI below 1 |
| (d) pooled radix-on hit rate, turns 2-5 | 0.709 | above 0 |

SGLang 0.5.16 (`lmsysorg/sglang:v0.5.16-cu130`) against vLLM 0.30.0 (the
vllm-devai backend), both at 196,608 tokens, 20 salted five-turn sessions per
configuration, MTP and reasoning off. vLLM's own prefix cache served 0.640 of
the same turns' prompt tokens, so both engines reuse the history; SGLang's
radix tree buys about 19% on later turns, not the 25% I16 asks for. Report:
`/var/cache/devai/bench/phase15/keep-check.json` (host-local); the check
itself is in git at commit d507d6b (`scripts/bench/sglang_keep_check.py`),
removed in 971b8e1.

## What was moved here

```
sglang/
  scripts/probe-sglang-reasoning.py        the SGLang prober (fit + load passes)
  tests/python/test_picker_sglang.py       the picker's SGLang rows and routing
  tests/python/test_bench_sglang_gate.py   the bench's SGLang candidacy gate
  tests/test-router-sglang.sh              `make test-sglang`
  tests/test-probe-sglang.sh               `make test-probe-sglang`
  Makefile.frozen-targets                  probe-sglang, probe-load-sglang,
                                           bench-sglang, test-sglang,
                                           test-probe-sglang, verbatim
  frozen-branches.txt                      everything cut out of live files:
                                           the compose service and the router's
                                           SGLANG_* wiring, the router's backend
                                           entry, the picker's entries, the
                                           router-sglang provider in each agent
                                           seed, test-router.sh's Tests 3 and 6,
                                           the SGLang cases of shared test files,
                                           the script and Makefile one-liners
```

## What stays live, unreachable

Nothing serves port 11436 and no picker row is SGLang, but this code stays in
place, tested where it was tested, so a thaw is a re-wiring, not a rewrite:

- gpu-arbiter: `sglangEntrypoint` (now with `--enable-metrics`), the SGLang
  branches of the reasoning policy, the Anthropic `/v1/messages` and
  Responses rewrites, engine-cache volumes, recovery-flag scoping and the
  argv golden cases.
- scripts: `_probe_hf_common`'s SGLang handling, the bench runner's SGLang
  tables and metric patterns, `select-models.py`'s `sglang_weight_gaps`, the
  card-hint and recipe tables, `deploy/backend-flags.yaml`'s `sglang:` section
  (`make verify-backend-flags` still checks the image), and the picker's
  Codex gate (`_AGENT_BACKEND_FAILS`: Codex is never offered on SGLang).
- data: `deploy/.sglang-reasoning-cache.json` (16 of its 21 rows were
  measured on an older SGLang image), and the SGLang store
  `/var/cache/devai/sglang` (Gemma-4-26B-A4B-it-NVFP4, gpt-oss-20b,
  NVIDIA-Nemotron-Nano-9B-v2-NVFP4, Ornith-1.0-9B-NVFP4, Qwen3.5-9B-NVFP4;
  mostly hard links into the vLLM store, so removing it frees little).

## To restore

1. Put back the cut branches from `frozen-branches.txt`, in this order: the
   router's backend entry, `SGLANG_URL`, the probe-cache load and the drift-map
   entry (`gpu-arbiter/main.go`, and its test `TestVLLMDevai_DriftCheckReadsItsOwnProbeCache`);
   the compose service and the router's wiring; `CACHE_SERVICES` /
   `CACHE_BACKEND_SERVICES` and the test in `test_bench_store_gap.py` that
   pins them; the picker's `_BACKENDS` and `_PICKER_BACKENDS`; the
   `router-sglang` providers in `config/`; the script one-liners; and remove
   the early `return` in `select-models.py`'s `enforce_sglang_store_gaps`.
2. Move the prober and the tests back from here, and paste the targets from
   `Makefile.frozen-targets` into the Makefile (with their `.PHONY` entries,
   `bench:` and `test:`). Restore the SGLang cases in `test_needle_warning.py`,
   `test_probe_classify.py`, `test_probe_scratch_ledger.py` and
   `gpu-arbiter/flags_allowlist_test.go`, and the expectations that dropped
   `sglang` / `router-sglang` (test_picker_opencode, test_picker_pi,
   test_model_sync, test_probe_exact_ctx, test_vllm_recipes,
   test_vllm_devai_backend).
3. Re-probe before offering anything: the cache predates the freeze.
   `make cache-down && make probe-sglang && make probe-load-sglang`.
4. The platform rebuild changes the ground under all of this: from Phase 4 on
   there is no stock SGLang image, only `devai-engines` compiled here, and
   Phase 4 skipped SGLang's stage. A thaw after Phase 4 means adding that
   stage (steps 4a-4b as the plan wrote them) before step 1 above.
5. Decide against I16 again, on the then-current vLLM: a newer SGLang that
   meets (b) is the reason to thaw, not the code being here.

## Open questions carried from the SGLang plan

From `docs/plans/sglang-backend-remediation.md` (removed 2026-10-05; git
history, commit ef4616a^), with what has answered them since:

1. Was the keep-or-freeze question open? -- Yes: Phase 15 asked it first and
   answered it (M16).
2. The adjudication thresholds -- written down before the runs as I16, and
   applied as written.
3. The unexplained Qwen3.5-9B-NVFP4 HumanEval result, 0.86 on SGLang against
   0.68 on vLLM on the same checkpoint (July 2026). Still unexplained and not
   re-measured: those rows are no longer in the bench cache, and they predate
   the bench's 2026-09-27 sampling and scorer fixes. The plan's own advice
   stands: it should not alone carry a keep.
4. A multi-hour GPU window for a fleet re-probe -- moot while frozen; step 3
   above needs one.
