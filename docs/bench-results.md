# Bench results -- vLLM leaderboard on RTX PRO 4000 Blackwell (24 GB)

> **How to read this page.**
>
> - **One run per cell.** Re-runs of one configuration change the
>   outcome of 5-23 % of items, depending on task and model ("Run-to-run
>   variation").
> - **Stochastic decoding at each engine's default sampler**
>   (temperature 0.6-1.0 by model). The harness's `temperature=0` never
>   reached the backend (defect D1).
> - **Subsets.** GSM8K and HumanEval use the *first n* items (a fixed,
>   non-random set: scores describe those items); MMLU-Pro and GPQA use
>   one seeded random sample shared by all models.
> - **Time-outs are scored wrong** (D2; 600 s for GSM8K and tools_use,
>   900 s otherwise, queueing included). Column t counts them; a score
>   means "correct within the time limit under the harness's queueing".
> - **TPS is characters/4**, one median from one run (D4).
> - **The HumanEval extractor can fail correct fenced function bodies**
>   (D3).
> - Intervals are 95 % Clopper-Pearson unless named; comparisons are
>   paired on identical items. Methods: [statistics-primer.md](statistics-primer.md)
>   (Sec. 1-3 estimates and intervals, 4-6 comparisons and multiple
>   testing, 7 designs that void the error rate).

> Measured 2026-05-05 (host_env_id `ea4fd7e7b668`: kernel
> `6.12.85+deb13-amd64`, driver `595.71.05`, GPU `NVIDIA RTX PRO 4000
> Blackwell`, CUDA `13.2`), vLLM via the gpu-arbiter router (port 11435;
> image tag `latest-cu130-ubuntu2404` as stated at the time, not
> re-verifiable), harness `scripts/bench/` with inspect_ai 0.3.158 (see
> [router.md](router.md) "Benchmark harness").
>
> **Frozen snapshot.** The 2026-05-05 rows are no longer in
> `deploy/.bench-cache.json` or any retained copy. This page is rebuilt
> from the eval logs in `/var/cache/devai/bench/inspect-logs/` (every
> quality number reproduces exactly) and the run log
> `/var/cache/devai/bench/bench-vllm-run-20260505T140328Z.log`. The
> picker reads the *current* cache, whose rows for these models were
> re-measured from July 2026 at other contexts and with more tasks
> (`make bench-report`; analysis in "Reproducing the statistics"). The
> cache schema is v3 since 2026-05-15; `BENCH_FORCE=1` re-runs tasks but
> does not reset a row (`update_row` is a pure merge).

## TL;DR

On the 2026-05-05 sweep, nine vLLM rows were each run once, at the
engine-default sampler, on GSM8K (first 100 items), HumanEval (first 50)
and tools_use (all 20 hand-written prompts). The data separate them only
coarsely:

- **Two rows are lower than every other row.** R1-Distill-Llama-8B
  (aggregate 0.477) and Nemotron-Nano-9B-v2 (0.267) sit below each of
  the other seven: raw p <= 0.0032, Holm-adjusted p <= 0.048 (paired
  Monte Carlo sign-flip test on the aggregate, family of 36 pairs;
  [statistics-primer.md](statistics-primer.md) Sec. 4.5, 6).
- **No difference among the top four is established** after
  correcting for the 36 pairwise comparisons: Nemotron-3-Nano (0.990),
  Qwen3-14B (0.973), Qwen3-8B (0.933), gpt-oss-20b (0.933). Their
  largest gap, Nemotron-3-Nano vs Qwen3-8B (+0.057, bootstrap 95% CI
  +0.013 to +0.110), has raw p = 0.024 and Holm-adjusted p = 0.26. The
  highest point estimate, Nemotron-3-Nano, exceeds Qwen3-14B by +0.017
  (95% CI -0.010 to +0.047, raw p = 0.38).
- **Qwen3.5-9B (0.903)** differs from Nemotron-3-Nano (raw p = 0.0016,
  Holm p = 0.027) and from the bottom two rows; no other difference is
  established. Its HumanEval shortfall is partly a scorer artifact
  (defect D3).
- **R1-Distill-Qwen-7B (0.803) and Llama-3.1-8B (0.760)** are below
  Nemotron-3-Nano and Qwen3-14B, and Llama-3.1-8B also below Qwen3-8B
  and gpt-oss-20b: six pairs with raw p <= 0.0024 and Holm p <= 0.039.
- **Tier assignments are not robust.** For 6 of the 9 rows, the 95%
  interval of the tools_use score contains the 0.9 badge threshold.
  gpt-oss-20b (17/20) and Qwen3-8B (19/20) fall on opposite sides of
  the threshold by two items.
- **Two failures probably trace to the serving setup.**
  R1-Distill-Llama-8B scored 0/50 on HumanEval, and 49 of its 50
  answers contain raw byte-level BPE markers (Issue #1; the mechanism is
  suspected, not confirmed). Nemotron-Nano-9B-v2 had no tool or
  reasoning parser configured in 2026-05; after the parsers were wired
  it scored HumanEval 40/50 and tools_use 20/20 (2026-07-20, ctx 131072,
  single runs, other conditions also changed).
- **Speed is one run per model** (chars/4 TPS, n = 1 run). The two MoE
  checkpoints had the highest medians (143.8 and 139.2 tok/s). No
  interval can be computed for a single-run median. Repeat runs of the
  same model varied with a CV of about 1.3 % within days (95% CI
  roughly 0.8-2.6 %, 8 model pairs); across months, images or contexts the max/min ratio was
  1.03-1.13 (per-model CVs 1.6-6.2 %, confounded by context).

There is no evidence here for an "outright leader" or a "clear
ordering" among the top rows. "Production default" below means "passed
fixed thresholds on point estimates in one run", with the caveats listed.

## Results: 2026-05-05 sweep

Standard layout ([statistics-primer.md](statistics-primer.md) Sec. 14):
n = items scored, x = successes, t = samples cut off by the time limit
(scored wrong), CP 95% = Clopper-Pearson interval for x/n, and "If t
correct" = the range x/n to (x+t)/n, the score if none or all of the
cut-off samples had been answered correctly; with sampling uncertainty
the conservative 95% interval for that range is [CP-lower(x, n),
CP-upper(x+t, n)], given in brackets ([statistics-primer.md](statistics-primer.md)
Sec. 3, partial identification). Common conditions: one run;
engine-default stochastic decoding ("Methodology"); vLLM with 10
samples in flight; GSM8K first 100 of 1319 test items, HumanEval first
50 of 164, tools_use all 20 prompts; runs 2026-05-05 14:06-17:11 UTC.

| Model (served name) | Task | n | x | t | x/n | CP 95% | If t correct | Notes |
|---|---|---:|---:|---:|---:|---|---|---|
| NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4@131072 | GSM8K | 100 | 99 | 0 | 0.99 | [0.946, 1.000] | - | |
| | HumanEval | 50 | 49 | 0 | 0.98 | [0.894, 0.999] | - | |
| | tools_use | 20 | 20 | 0 | 1.00 | [0.832, 1.000] | - | subcases E/S/M/F 5/5/5/5 |
| Qwen3-14B-NVFP4@65536 | GSM8K | 100 | 98 | 0 | 0.98 | [0.930, 0.998] | - | |
| | HumanEval | 50 | 47 | 0 | 0.94 | [0.835, 0.987] | - | |
| | tools_use | 20 | 20 | 0 | 1.00 | [0.832, 1.000] | - | 5/5/5/5 |
| Qwen3-8B-NVFP4@131072 | GSM8K | 100 | 97 | 0 | 0.97 | [0.915, 0.994] | - | |
| | HumanEval | 50 | 44 | 0 | 0.88 | [0.757, 0.955] | - | |
| | tools_use | 20 | 19 | 0 | 0.95 | [0.751, 0.999] | - | 5/5/5/4 |
| gpt-oss-20b@262144 | GSM8K | 100 | 97 | 0 | 0.97 | [0.915, 0.994] | - | |
| | HumanEval | 50 | 49 | 0 | 0.98 | [0.894, 0.999] | - | |
| | tools_use | 20 | 17 | 0 | 0.85 | [0.621, 0.968] | - | 2/5/5/5; 3 empty_schema answers invented arguments |
| Qwen3.5-9B-NVFP4@131072 | GSM8K | 100 | 98 | 0 | 0.98 | [0.930, 0.998] | - | |
| | HumanEval | 50 | 39 | 0 | 0.78 | [0.640, 0.885] | - | 5 of 11 failures are extractor-defect candidates (D3) |
| | tools_use | 20 | 19 | 1 | 0.95 | [0.751, 0.999] | 0.95-1.00 [0.751, 1.000] | 4/5/5/5; the one failure is the time-out |
| DeepSeek-R1-Distill-Qwen-7B@65536 | GSM8K | 100 | 87 | 0 | 0.87 | [0.788, 0.929] | - | |
| | HumanEval | 50 | 47 | 1 | 0.94 | [0.835, 0.987] | 0.94-0.96 [0.835, 0.995] | |
| | tools_use | 20 | 12 | 5 | 0.60 | [0.361, 0.809] | 0.60-0.85 [0.361, 0.968] | 5/2/3/2; 5 of 8 failures are time-outs |
| Llama-3.1-8B-Instruct-NVFP4@131072 | GSM8K | 100 | 78 | 3 | 0.78 | [0.686, 0.857] | 0.78-0.81 [0.686, 0.882] | |
| | HumanEval | 50 | 40 | 0 | 0.80 | [0.663, 0.900] | - | |
| | tools_use | 20 | 14 | 0 | 0.70 | [0.457, 0.881] | - | 5/2/5/2 (wrong arguments) |
| DeepSeek-R1-Distill-Llama-8B@32768 | GSM8K | 100 | 63 | 0 | 0.63 | [0.528, 0.724] | - | |
| | HumanEval | 50 | 0 | 1 | 0.00 | [0.000, 0.071] | 0.00-0.02 [0.000, 0.106] | BPE-decode bug (^*) |
| | tools_use | 20 | 16 | 0 | 0.80 | [0.563, 0.943] | - | 5/3/5/3 |
| NVIDIA-Nemotron-Nano-9B-v2-NVFP4@65536 | GSM8K | 100 | 74 | 24 | 0.74 | [0.643, 0.823] | 0.74-0.98 [0.643, 0.998] | 74/76 = 0.97 among completed |
| | HumanEval | 50 | 3 | 2 | 0.06 | [0.013, 0.165] | 0.06-0.10 [0.013, 0.218] | no reasoning parser configured |
| | tools_use | 20 | 0 | 1 | 0.00 | [0.000, 0.168] | 0.00-0.05 [0.000, 0.249] | router stripped tools (^+) |

Subcases (5 prompts each): E = empty_schema, S = single_arg, M =
multi_tool_pick, F = result_followup. A subcase score of 5/5 has the 95%
interval [0.478, 1.000], and 2/5 has [0.053, 0.853]: subcase
differences in this table are descriptive only. "Accuracy among
completed" is biased upwards, because the items that time out tend to
be the hard ones.

**Aggregate.** The aggregate is the unweighted mean of the three
proportions above, whose n are 100, 50 and 20. It has no standard
interpretation. Its uncertainty is dominated by the tools_use term.
The intervals come from a stratified percentile bootstrap that
resamples items within each task (B = 10000, fixed seed). "Gap" is
the difference to the next row, with a paired bootstrap interval
(unadjusted for multiplicity) and the p of a paired sign-flip
randomization test, computed by Monte Carlo (10 000 random sign flips,
fixed seed, +1 correction, so the smallest attainable p is about
0.0001). The Holm adjustment runs over the 8 adjacent gaps; that
family is data-selected, because adjacency comes from the observed
ranks, so its adjusted p-values are optimistic. The all-pairs family of
36 below does not have that problem
([statistics-primer.md](statistics-primer.md) Sec. 4.1, 4.5, 6).

| Rank by point estimate | Model | Aggregate | 95% CI | Gap to next | Gap 95% CI | p | Holm p (8) |
|---:|---|---:|---|---:|---|---:|---:|
| 1 | Nemotron-3-Nano | 0.990 | [0.973, 1.000] | +0.017 | [-0.010, +0.047] | 0.38 | 1.00 |
| 2 | Qwen3-14B | 0.973 | [0.947, 0.993] | +0.040 | [0.000, +0.087] | 0.11 | 0.54 |
| 3 | Qwen3-8B | 0.933 | [0.883, 0.973] | 0.000 | [-0.070, +0.073] | 1.00 | 1.00 |
| 4 | gpt-oss-20b | 0.933 | [0.873, 0.983] | +0.030 | [-0.047, +0.103] | 0.53 | 1.00 |
| 5 | Qwen3.5-9B | 0.903 | [0.850, 0.950] | +0.100 | [+0.003, +0.197] | 0.084 | 0.50 |
| 6 | R1-Distill-Qwen-7B | 0.803 | [0.723, 0.880] | +0.043 | [-0.047, +0.133] | 0.41 | 1.00 |
| 7 | Llama-3.1-8B | 0.760 | [0.677, 0.840] | +0.283 | [+0.213, +0.343] | < 0.0002 (MC floor) | < 0.002 |
| 8 | R1-Distill-Llama-8B | 0.477 | [0.407, 0.540] | +0.210 | [+0.133, +0.283] | 0.0032 | 0.022 |
| 9 | Nemotron-Nano-9B-v2 | 0.267 | [0.230, 0.303] | | | | |

The randomization test is the decision criterion. The bootstrap
interval is an approximate companion. For rank 5 vs 6 the two disagree
at the margin: the interval excludes 0 while p = 0.084. A difference not
established here is not shown to be absent: for "no difference" claims
the raw p and the interval are the relevant numbers, because a
multiplicity adjustment only makes non-rejection easier
([statistics-primer.md](statistics-primer.md) Sec. 6).

**All 36 pairs.** With Holm over all 36 pairwise aggregate comparisons
(Monte Carlo p; 17 raw p-values sit at the floor, reported as p <
0.0002), 22 pairs differ at the 5% level. No difference among the top
four is established after this correction: the smallest raw p among
them is 0.024 (Nemotron-3-Nano vs Qwen3-8B, +0.057, bootstrap 95% CI
+0.013 to +0.110), Holm-adjusted 0.26. Qwen3.5-9B differs from
Nemotron-3-Nano (+0.087, 95% CI +0.037 to +0.143, raw p = 0.0016,
adjusted p = 0.027) and from the bottom two. R1-Distill-Qwen-7B
differs from Nemotron-3-Nano and Qwen3-14B; Llama-3.1-8B from those two
and from Qwen3-8B and gpt-oss-20b. The bottom two differ from every row,
including each other. An independent exact sign-flip
computation gave Holm-adjusted p-values of 0.029, 0.030 and 0.041 for
the three pairs nearest the 5% line (Monte Carlo: 0.027, 0.038, 0.048);
the conclusions do not change. Per task (exact McNemar, raw p >= 0.125), the
highest-scoring row, Nemotron-3-Nano, is not distinguishable from
Qwen3-14B, Qwen3-8B and gpt-oss-20b on any of the three tasks, nor from
Qwen3.5-9B on GSM8K and tools_use, R1-Distill-Qwen-7B on HumanEval, or
R1-Distill-Llama-8B on tools_use. Every pair, test and adjusted p is in
the output of `scripts/stats/bench_stats.py` ("Reproducing the
statistics").

**Practical vs statistical difference.** A gap that is "not
distinguishable" is not shown to be zero. At these sample sizes the
data cannot resolve gaps of a few points. For HumanEval at n = 50, the
95% half-width of an unpaired difference is 0.16 at a score of 0.8 and
0.12 at 0.9. For a paired difference with 10 % discordant items it is
0.09. Whether a 2-point gap matters is a product decision, and these
data cannot resolve a 2-point gap ([statistics-primer.md](statistics-primer.md)
Sec. 5).

### Speed, latency, VRAM and leak (same sweep, single values)

One run, no replication. **TPS**: median per-request decode rate over
the 40 latency prompts (characters/4 per second, defect D4; temperature
0). **Cold**: time to first token of the first prompt, a cold start only
if that request triggered the launch. **Warm p50/p95**: over the other
39 prompts (type-7 quantiles; at n = 39 the p95 lies between the 2nd-
and 3rd-largest values). **Peak VRAM**: device-wide `nvidia-smi`
maximum at 1 Hz, in GiB (the cache field is named `_gb`; values are
MiB/1024). **Leak**: template-marker regex matches over the 40
responses, a count, not a proportion of prompts.

| Model | Leak (matches / 40 prompts) | Cold (s) | Warm p50/p95 (ms) | TPS | Peak VRAM (GiB) |
|---|---|---:|---|---:|---:|
| Nemotron-3-Nano | 0 | 60.6 | 46.9/52.7 | 143.8 | 22.45 |
| Qwen3-14B | 0 | 47.8 | 46.1/47.3 | 61.9 | 22.32 |
| Qwen3-8B | 0 | 43.5 | 32.3/34.4 | 102.1 | 22.53 |
| gpt-oss-20b | 0 | 53.9 | 50.3/54.0 | 139.2 | 22.37 |
| Qwen3.5-9B | 0 | 160.5 | 31.5/45.6 | 55.3 | 21.54 |
| R1-Distill-Qwen-7B | 0 | 84.8 | 36.2/49.4 | 44.5 | 21.90 |
| Llama-3.1-8B | 0 | 39.5 | 23.3/24.0 | 95.4 | 22.65 |
| R1-Distill-Llama-8B | 0 | 0.05 (warm, ^#) | 37.1/52.7 | 42.5 | 21.69 |
| Nemotron-Nano-9B-v2 | 3 `</think>` | 112.7 | 26.9/30.9 | 80.4 | 22.28 |

Leak: the probe is a fixed hand-written set of 40 prompts decoded at
temperature 0, so the measured count is close to a fixed property of
those prompts. An interval says something only under the assumption
that the 40 prompts are exchangeable with a wider population of
similar prompts. Under that assumption, 0 matches in 40 prompts bounds
a per-prompt leak probability at 0.088 (upper 95% CP limit), and
Nemotron-Nano-9B-v2's 3 matches are 0.075 matches per prompt (Garwood
exact Poisson interval 0.015 to 0.22). 1 to 3 prompts were affected;
the number was not recorded.

### Footnotes

- ^* **HumanEval 0/50 for R1-Distill-Llama-8B is the byte-level
  BPE-decode bug** (Issue #1). The U+0120 space and U+010A newline
  markers appear un-decoded in 49 of 50 answer texts on 2026-05-05
  (50/50 and 49/50 in the two 2026-05-02 runs), so the code cannot
  run. Three runs, all 0/50: CP [0.000, 0.071] each.
- ^+ **tools_use 0/20 for Nemotron-Nano-9B-v2**: all 20 samples failed
  with "no tool call". The router's `maybeStripTools` removes `tools`
  and `tool_choice` when no tool parser is probed, and in 2026-05 this
  model had none. The score measures the stripping, not the model.
  Since 2026-07-28 the harness records such rows as skipped instead of
  0.0.
- ^! **HumanEval 47/50 for R1-Distill-Qwen-7B vs 13/50 on 2026-05-02.**
  Same 50 items (verified by content hashes), paired: b = 34 items
  passed only on 2026-05-05, c = 0 only on 2026-05-02; exact McNemar
  p = 1.2e-10, difference +0.68 (95% CI +0.52 to +0.78). The scorer
  changed between the runs (v1 to v2 cleaner, Issue #4), and the
  retained completions support that as the cause: all 37 failures on
  2026-05-02 were syntax errors on the model's prose preamble ("To
  solve this problem, ..."), all 37 programs re-extracted with the v2
  cleaner parse (a parse-only check; no code was run), and 34 of those
  37 items passed on 2026-05-05. Because every answer was regenerated,
  decoding randomness cannot be excluded item by item. The backend
  image was not recorded.
- ^# **The cold start of 0.05 s for R1-Distill-Llama-8B is not a cold
  start.** The container was already warm when the sweep began (the
  run log reads `ttft_first=51.1ms`). Router launch-to-ready times for
  this model in 2026-05 had median 68 s (n = 9 distinct launches;
  "Cold-start signal"). The earlier "85.2 s from the 2026-05-02 sweep"
  is not retained.
- The Nemotron-3-Nano row is from the run after commit b730985
  (2026-05-05 09:36 UTC), which dropped `--enforce-eager` and added
  `--max-num-seqs 8`. The fix did *not* land before the 2026-05-02
  sweep, which had no Nemotron-3-Nano row. A pre-fix run at 09:17 UTC
  scored GSM8K 98/100, HumanEval 43/50 and tools_use 20/20. The
  HumanEval change 43 -> 49 is not distinguishable at the 5% level
  (b = 7, c = 1, exact McNemar p = 0.070).

### Run-to-run variation

Several (model, task) cells were re-run, incidentally, with the same
served name, items and harness settings. They show how much a single
run moves ([statistics-primer.md](statistics-primer.md) Sec. 4.4).

**Fraction of items that changed outcome between two runs**, pooled
over re-runs with no documented configuration change in between. The
pools mix models, and discordance scales with p(1-p), so a
high-scoring model changes fewer items than a mid-scoring one; read the
figures per task and model, not as one rate:

| Task | Fraction changed | Item pairs | Re-run pairs behind it |
|---|---:|---:|---|
| GSM8K | 0.096 | 1000 | 10 (the eight 2026-05-02 vs 05-05 pairs, 0.01-0.30; Gemma-4-26B; gpt-oss-20b on SGLang) |
| HumanEval | 0.05 (0.067 without R1-Llama's degenerate 0/50 vs 0/50 pair) | 200 | 4 |
| HumanEval+ | 0.08 | 150 | 3 (incl. gpt-oss-20b 42 -> 47/50, 22 minutes apart) |
| MMLU-Pro | 0.19 | 100 | 1 (gpt-oss-20b on SGLang) |
| GPQA | 0.22 | 60 | 1 (qwen3.6:35b, two back-to-back runs in one container) |
| tools_use | 0.23 | 60 | 3 (R1-Distill-Qwen-7B, R1-Distill-Llama-8B, Llama-3.1-8B: low-scoring models only) |

Examples: gpt-oss-20b scored HumanEval+ 42/50 and then 47/50 22
minutes later (b = 2, c = 7, raw p = 0.18). GSM8K re-run three days apart
(2026-05-02 vs 2026-05-05, same context) moved Qwen3-14B 97 -> 98,
Qwen3-8B 98 -> 97, gpt-oss-20b 97 -> 97, Qwen3.5-9B 95 -> 98,
R1-Distill-Qwen-7B 88 -> 87, Llama-3.1-8B 84 -> 78, R1-Distill-Llama-8B
57 -> 63 and Nemotron-Nano-9B-v2 84 -> 74. None of these changes is
distinguishable at the 5% level; the smallest p is 0.099, for
Nemotron-Nano-9B-v2, whose time-outs rose from 14 to 24.

**Consequence for intervals.** In the single clean re-run pair
available for each, decoding randomness accounts for 0.47 of the
binomial variance on MMLU-Pro and 0.59 on GPQA (one pair each, so no
uncertainty can be given). The finite-population correction therefore
does not apply in full, and the uncorrected Clopper-Pearson interval is
the one reported ([statistics-primer.md](statistics-primer.md) Sec. 3,
"Finite-population correction").

## Methodology

Per (model, backend, ctx) pair the harness (`scripts/bench/bench_runner.py`)
runs, in order:

1. **Latency/leak sidecar** (`bench_latency_leak.py`): the first 40 of
   the 42 prompts in `data/latency_prompts.jsonl`, streamed one at a
   time with `temperature: 0` and `max_tokens` 16-256. The first
   prompt's time to first token is `ttft_ms_first`; the other 39 feed
   `ttft_ms_steady_p50/p95`. Each request's decode rate is
   `tokens / (t_done - t_first_token)` and `tps_sustained_p50` is their
   median, first prompt included; counting N tokens over N-1
   inter-token intervals biases a rate upwards by about 6.7 % at 16
   tokens and 0.4 % at 256. The response bodies, `reasoning_content`
   included, are regex-swept for the markers in `data/leak_markers.txt`;
   `leak_rate` = total matches / 40.
2. **GSM8K** (inspect_ai): first 100 of the 1319 test items,
   `shuffle=False`. The scorer takes the number after `####`, else the
   last number, and tests numeric equality. Time limit 600 s.
3. **HumanEval**: first 50 of 164, `shuffle=False`; pass@1 from one
   generated sample per problem, run with its tests in a subprocess
   (10 s, memory cap). Time limit 900 s.
4. **tools_use**: all 20 prompts of `data/tools_prompts.jsonl` (5 per
   subcase), each scored 1 or 0: exactly one tool call, to the expected
   tool, with exactly the expected arguments; for result_followup the
   final text must also contain an expected substring. Time limit 600 s.
   The protocol changed twice: `tool_choice="auto"` through inspect's
   loop before 2026-05-02 about 18:50 UTC; pinned to the expected
   function from then to 2026-07-20 (so multi_tool_pick no longer tests
   routing); since 2026-07-20 auto or pinned per the model's probed
   `tool_mode`, recorded on the row. Scores from different protocols
   are not comparable.
5. **Added later** (not in this sweep): HumanEval+ (first 50, EvalPlus
   tests); MMLU-Pro and GPQA-Diamond, each `shuffle(seed=42)` then the
   first n of 12032 and of 198 (GPQA options shuffled per question with
   a seed derived from its text). n is 60 or 100 depending on the row;
   the smaller subset is a prefix of the larger.
6. **VRAM sampler**: device-wide `nvidia-smi` at 1 Hz from before the
   first request to the end of the run. The peak is a lower bound; the
   mean includes model load, so it depends on run length.

**Concurrency and time limits.** inspect_ai keeps 10 samples in flight
on vLLM and SGLang, and did on Ollama until 2026-09-19 (1 since). A
sample's time limit starts when inspect starts the sample, so queueing
counts against it; on Ollama's single slot one long generation could
hold nine others. Samples that time out are scored wrong (defect D2):
their outcomes are censored, and time-outs on a shared slot are not
independent ([statistics-primer.md](statistics-primer.md) Sec. 7,
"Informative truncation", "Dependent items").

**Decoding actually used.** No scored request carried a temperature,
top_p, top_k, seed or max_tokens (defect D1). vLLM and SGLang then use
the checkpoint's `generation_config.json` sampling keys if it has any,
otherwise temperature 1.0 and top_p 1.0. The engine logs (July 2026
onwards) record:

| Model | Default sampler (source) |
|---|---|
| Qwen3-8B-NVFP4, Qwen3-14B-NVFP4 | T 0.6, top_k 20, top_p 0.95 (vLLM log) |
| NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 | T 1.0, top_p 1.0 (vLLM log) |
| gpt-oss-20b, Qwen3.5-9B-NVFP4, Nemotron-Nano-9B-v2-NVFP4, Ornith-1.0-9B-NVFP4 | no sampling keys in `generation_config.json`: engine default T 1.0, top_p 1.0 |
| DeepSeek-R1-Distill-Qwen-7B | T 0.6, top_p 0.95 (SGLang log, 2026-04-30 to 2026-05-02) |
| Gemma-4-26B-A4B-it-NVFP4 | T 1.0, top_k 64, top_p 0.95 |
| Qwen3.8-27B builds (vLLM, vllm-devai) | T 1.0, top_k 20, top_p 0.95 |
| Ollama models | Modelfile defaults (qwen3.6:35b-a3b-mtp: T 1.0, top_k 20, top_p 1.0, presence_penalty 1.5) |

The vLLM engine log for 2026-05 is not retained; the 2026-05-05 rows
most likely ran at the same checkpoint defaults (unverified). So every
answer is a random draw; models ran under different samplers, and a gap
between two models mixes capability with sampler; and the greedy
latency/leak sidecar did not produce the same draws as the scored tasks.

**Single-run design.** One run per cell, no seed, no replication. The
only evidence on run-to-run variation is the incidental re-runs in
"Run-to-run variation".

**Statistics** ([statistics-primer.md](statistics-primer.md)).
Clopper-Pearson 95% for single proportions, the primary interval (Wilson
also in the script output; Sec. 3). For GPQA and MMLU-Pro a
finite-population-corrected interval is only a lower bound on width,
"valid only under deterministic decoding" (Sec. 3). First-n scores
describe those items (Sec. 1). For time-outs, the range [x/n, (x+t)/n]
and its conservative 95% version [CP-lower(x, n), CP-upper(x+t, n)]
(Sec. 3). Paired comparisons (the harness guarantees identical items)
use the exact McNemar test and Newcombe's method 10 (Sec. 4.1). The
aggregate uses a stratified percentile bootstrap and a paired sign-flip
randomization test by Monte Carlo (10 000 flips, fixed seed, +1
correction; Sec. 4.5). Families use Holm, which controls claims that a
difference exists; claims of "no difference established" are given with
the raw p and an interval (Sec. 6). Latency quantiles are type 7 (Sec.
8); no per-request samples are stored, so no median interval exists.

**Cache layout.** One row per (model, backend, ctx) in
`deploy/.bench-cache.json` (schema v3), keyed
`<repo>@<sha>::<backend>::<ctx>` (HF) or `<digest>::<backend>::<ctx>`
(Ollama). Every task entry stamps `host_env_id`, `n` and `ran_at`; the
inspect log behind an entry is found by (task, n, completion time), and
all 131 current entries match their log exactly.

### Open harness defects (documented, not fixed)

- **D1 -- sampling is never applied.** `_invoke_inspect_task` passes
  `config=GenerateConfig(...)` to `inspect_ai.eval()`, which has no
  `config` parameter: it takes generation settings as plain keywords
  and builds `GenerateConfig(**kwargs)`, a pydantic model that ignores
  unknown fields. Evidence: `model_generate_config` is `{}` in every
  retained eval log and none of 16,565 logged request bodies carries
  temperature, top_p, top_k, seed or min_p, while `max_connections=1`,
  passed as a keyword, does appear. The `sampling: {temperature: 0.0,
  source: greedy_default}` stamp on 8 cache rows is false, and
  `deploy/bench-sampling.json` overrides never took effect.
- **D2 -- time-outs are scored wrong.** A sample over its time limit is
  recorded incorrect with an empty answer. 65 eval logs contain such
  samples; 33 of 131 current cache task entries have at least one (e.g.
  Nemotron-3-Nano GPQA 48/100 with 40 time-outs, 0.80 among completed;
  qwen3.5:27b GSM8K 60/100 with 40; Ornith-1.0-9B GPQA on SGLang 63/100
  with 33). Listing: `scripts/stats/bench_timeouts.py`.
- **D3 -- the HumanEval extractor strips the first line's indentation.**
  In `_FENCE_BLOCK_RX` (```` ```(?:python|py)?\s*\n?(.*?)``` ````) the
  `\s*` consumes the newline and the first code line's leading spaces,
  and `\n?` then matches nothing. A fenced *function body* fails with
  IndentationError or "'return' outside function"; a fenced full
  definition is unaffected. A parse-only scan of all 91 v2-scored
  HumanEval/HumanEval+ runs found 30 failures that would parse under a
  corrected extractor, in 10 runs of Qwen3.5-9B, Ornith-1.0-9B and
  qwen3.6:35b; whether they would pass the tests is unknown without
  executing them (`scripts/stats/bench_scorer_scan.py`).
- **D4 -- TPS is characters/4.** The stream parser uses
  `usage.completion_tokens` only if the engine sends it, and the bench
  never sets `stream_options.include_usage`. The engine sources checked
  (vLLM 0.28 and Ollama v0.34.2) send no usage without it, so there
  `effective_tokens` = `chars // 4`, before and after commit 5832174,
  which is inert for the bench. For SGLang and stock vLLM 0.22.1 the
  same is inferred from the cached values, not verified in source.
  The chars/4 estimate was checked only as a summary, not per request:
  the bench's median rate (client timing, all 40 prompts) was 0.99-1.02
  of the engine-counted median over outputs of at least 128 tokens, and
  0.93-1.01 of the engine median over all prompts, in the same four runs
  (Qwen3.8 tokenizer). The harness stores no per-request character
  counts. The 10-25 % shortfall on code reported in commit 5832174 is not
  retained and was not reproduced.

## TPS counting fix

History (2026-05-02). The first bench pass reported TPS values 2.4-94x
lower than the later ones for models with a reasoning parser (for
example, 0.66 tok/s for Qwen3-14B-NVFP4). The fix in
`_bench_core.stream_chat_completion` accumulates the `reasoning_content`
characters alongside `content`, and computes
`effective_tokens = max(usage.completion_tokens, (content_chars +
reasoning_chars) // 4)`.

What the retained data say about this:

- **What the fix really does.** In the bench's streams
  `completion_tokens` is always 0 (defect D4), so the fix reduces to
  "count reasoning characters as well as content characters". The
  stated cause, that "vLLM's `usage.completion_tokens` only counts
  content tokens", is unverified: the streams carried no usage block at
  all. The pre-fix code was never committed, so the cause cannot be
  checked; a plausible (unverified) cause is that the first pass
  counted only content characters.
- **The before/after table below is historical and not reproducible.**
  The validation run was not retained. The "after" column is itself a
  single run, a different one from the 2026-05-05 sweep.

| Model | Parser | Before fix | After fix (2026-05-02) | 2026-05-05 sweep |
|---|---|---:|---:|---:|
| Qwen3-14B-NVFP4 | qwen3 | 0.66 | 62.02 | 61.94 |
| gpt-oss-20b | openai_gptoss (harmony) | 38.4 | 136.37 | 139.2 |
| Qwen3-8B-NVFP4 | qwen3 | 14.53 | 98.31 | 102.09 |
| Qwen3.5-9B-NVFP4 | qwen3 | 22.67 | 55.73 | 55.27 |
| DeepSeek-R1-Distill-Qwen-7B | deepseek_r1 | 13.85 | 45.18 | 44.47 |
| DeepSeek-R1-Distill-Llama-8B | deepseek_r1 | 17.51 | 42.51 | 42.48 |
| Nemotron-Nano-9B-v2-NVFP4 | (none, inline `<think>`) | 82.0 | 81.08 | 80.38 |
| Llama-3.1-8B-Instruct-NVFP4 | (none) | 95.53 | 95.88 | 95.43 |

The two rows without a separate reasoning channel changed by -1.1 % and
+0.4 %. Llama-3.1 is a non-reasoning model. Nemotron-Nano-9B-v2 does
reason, but inline in `content`, so the fix does not change its count.
Across all eight models, the three-day repeat (2026-05-02 values as
recorded here, 2026-05-05 from the run log) gives a single-run
coefficient of variation of about 1.3 % (SD of the log ratio 0.018; 95%
CI roughly 0.8-2.6 %, 8 model pairs, partly from unretained values);
changes of this size are within the observed run-to-run variation.
Across months, images or contexts the spread is larger: max/min ratios
of 1.03-1.13 over three or four runs per model (per-model CVs 1.6-6.2 %;
the largest, Qwen3-8B at 1.13 and Qwen3-14B at 1.08, are confounded by
a change of context length).

## Architectural finding: FP4 quantization + MoE on Blackwell

The single-run medians of the 2026-05-05 sweep (chars/4) group by
architecture. This is descriptive: each class holds one or two models
that differ in more than architecture, each value is one run, and no
causal claim follows.

| Architecture x quantization | Models | TPS (one run each) | Notes |
|---|---|---:|---|
| Hybrid Mamba-2/attention MoE + NVFP4 (~3 B active) | Nemotron-3-Nano-30B-A3B-NVFP4 | 143.8 | FP8 KV cache in the July launches; the 2026-05-05 launch passed no KV-dtype flag |
| MoE + MXFP4 (~3.6 B active) | gpt-oss-20b | 139.2 | |
| Dense + NVFP4 (8 B) | Qwen3-8B-NVFP4, Llama-3.1-8B-Instruct-NVFP4 | 102.1, 95.4 | |
| Hybrid Mamba-2/attention + NVFP4 (9 B, 4 of 56 layers attention; inline reasoning) | Nemotron-Nano-9B-v2-NVFP4 | 80.4 | |
| Dense + NVFP4 (14 B) | Qwen3-14B-NVFP4 | 61.9 | |
| Hybrid linear attention + NVFP4 | Qwen3.5-9B-NVFP4 (`ykarout/`) | 55.3 | gated-delta linear attention, not Mamba; BF16 lm_head over a 248K vocabulary |
| Dense + BF16 (7-8 B) | DeepSeek-R1-Distill-{Qwen-7B, Llama-8B} | 44.5, 42.5 | |

**Decode-ceiling model** ([statistics-primer.md](statistics-primer.md)
Sec. 9, "Model-based ceilings"). At batch 1 decode is bounded by memory
bandwidth / bytes read per token. These docs use 640 GB/s, for which no
primary source was found (672 GB/s is also quoted; neither is
verified). Bytes per token from safetensors headers or public parameter
counts: Qwen3-8B-NVFP4 5.15 GB (ceiling 124 tok/s at 640 GB/s; measured
98.3-110.9, utilisation 0.79-0.89); Qwen3.5-9B-NVFP4 8.93 GB (ceiling
71.7; utilisation 0.77-0.80); R1-Distill-Llama-8B BF16 15.0 GB, not 16
(ceiling 42.6; the measured 42.5 would be 99.7 % utilisation, which is
implausible: either the chars/4 token count is biased upwards for this
model, or the real bandwidth is higher than 640 GB/s; the data do not
separate the two).
The earlier "real-world ~15 tok/s after parser, framework, and
streaming overhead" contradicted this page's own 42.5 and is removed. A
better test uses engine-counted rates: two Qwen3.8-27B builds on
vllm-devai (2026-09-22, MTP off) are predicted to differ by their
bytes-per-token ratio, 1.080; the observed paired ratio is 1.069 (95%
CI 1.067 to 1.071). The prediction lies outside that interval: bytes
per token explain most of the difference, not all, and the pure bytes
model is rejected at face value. The interval covers prompts within one
run per build (pseudo-replication), so it understates the run-to-run
uncertainty. The MoE rows' per-token transfers (~1.5 and ~1.8
GB) are the page's earlier estimates, not re-derived, so "MoE + FP4
wins decisively" is a hypothesis consistent with the ceiling model, not
a measured effect (one run per model, no controlled comparison).
Nemotron-3-Nano: the retained evidence is one run per mode, 143.8
tok/s with CUDA graphs (2026-05-05 sweep, ctx 131072) and 40.25 with
`--enforce-eager` (current cache row, 2026-07-17, ctx 163840), a ratio
of about 3.6 confounded by context, `--max-num-seqs` and date. The
figures 42.87 and 144.84 quoted in followup #11 come from runs that were
not retained.

**Wall time.** The following table is arithmetic, not a measurement:
(assumed reasoning tokens + a 200-token answer) / TPS. The reasoning
lengths are assumptions and were not measured.

| Model | TPS | Assumed reasoning tokens | Computed wall time |
|---|---:|---:|---:|
| Nemotron-3-Nano-30B-A3B-NVFP4 | 144 | ~300 | 3.5 s |
| gpt-oss-20b | 139 | ~300 | 3.6 s |
| Qwen3-8B-NVFP4 | 102 | ~500 | 6.9 s |
| Llama-3.1-8B-Instruct-NVFP4 | 95 | 0 | 2.1 s |
| Qwen3-14B-NVFP4 | 62 | ~600 | 12.9 s |
| DeepSeek-R1-Distill-Qwen-7B | 45 | ~1500 | 37.8 s |

The qualitative point stands as arithmetic: for an interactive agent,
the length of the reasoning preamble can matter more than the raw decode
rate. The R1-Distill rows decode at 42-45 tok/s, but a long `<think>`
block makes each turn slow.

**Routing implication (hedged).** Among the rows measured, the
NVFP4/MXFP4 checkpoints of Nemotron-3-Nano, the Qwen3 8B/14B models and
gpt-oss-20b combine the highest quality point estimates with medians of
62-144 tok/s; no quality difference among these four is established
after the 36-comparison correction (see "Results"). The R1-Distill BF16
models score lower than Nemotron-3-Nano on GSM8K (paired exact McNemar:
R1-Distill-Qwen-7B raw p = 0.0018, Holm(24) p = 0.032;
R1-Distill-Llama-8B p < 1e-6). Their lower tools_use scores are point
estimates only (R1-Distill-Qwen-7B Holm(24) p = 0.12, R1-Distill-Llama-8B
raw p = 0.125). R1-Distill-Llama-8B is broken on HumanEval (Issue #1).

## Cold-start signal

`ttft_ms_first` is the TTFT of the first latency prompt. It includes
the container launch only if that request triggered one; in 3 of 27
current cache rows it did not (36.4, 68.0 and 373.8 ms). A single value
per model says little about variability. The router log records every
launch from "starting" to "ready" (2026-04-28 to 2026-09-26), with times
quantised by the router's 2 s `/health` poll.

**The persisted log repeats whole blocks of lines.** The logger sidecar
re-appends history, mostly in 2026-05. So it holds 3,245 "starting"
lines but only 742 distinct launches, which is what `perf_coldstart.py`
counts. Of the 742:

- **673** reached "ready";
- **11** never did: 10 real launches (9 hit a 10-minute health timeout,
  and 1, on 2026-04-30, a 5-minute one), and 1 request that named a
  client model;
- **36** engine failures;
- **7** superseded by a new start;
- **15** abandoned by a router restart.

The two never-ready launches on 2026-05-05 were
Qwen3-Coder-30B-A3B-Instruct-FP4 at ctx 65536; gpt-oss-20b had one, on
2026-05-01. An earlier version of this page counted the duplicated
lines, which inflated every n below and collapsed the intervals.
Medians and ranges for the 2026-05 configurations, distinct launches:

| Model | `ttft_ms_first` 2026-05-05 (s, one request) | Router launch-to-ready median (s) | 95% CI, assuming exchangeable launches | n | Range |
|---|---:|---:|---|---:|---|
| Llama-3.1-8B-Instruct-NVFP4 | 39.5 | 40 | [39, 48] | 35 | 38-50 |
| Qwen3-8B-NVFP4 | 43.5 | 44.5 | [43, 52] | 16 | 42-54 |
| Qwen3-14B-NVFP4 | 47.8 | 55 | [48, 58] | 10 | 47-58 |
| gpt-oss-20b | 53.9 | 56 | [53, 62] | 8 | 53-62 |
| Nemotron-3-Nano-30B-A3B-NVFP4 | 60.6 | 40 | [36, 46] | 10 | 28-48 (launch flags changed in this window) |
| DeepSeek-R1-Distill-Llama-8B | 0.05 (warm) | 68 | [67, 74] | 9 | 66-75 |
| DeepSeek-R1-Distill-Qwen-7B | 84.8 | 69 | [67, 76] | 10 | 64-77 |
| Nemotron-Nano-9B-v2-NVFP4 | 112.7 | 77 | none at n = 5 | 5 | 75-88 |
| Qwen3.5-9B-NVFP4 | 160.5 | 143 | [141, 152] | 10 | 140-153 |

The two columns measure different things. `ttft_ms_first` is one
launch plus the first request's prefill, in the sweep. The router
medians cover every launch of that model in the window, whatever
triggered it, and the launch flags were not constant throughout
(Nemotron-3-Nano's changed on 2026-05-05). The order-statistic
intervals assume exchangeable launches; consecutive launches are likely
serially dependent (for example through a warm page cache) and flags
changed, so treat the intervals as descriptive
([statistics-primer.md](statistics-primer.md) Sec. 8).

- **"Fastest cold start".** Llama-3.1-8B and Nemotron-3-Nano (median
  40 s each, n = 35 and 10) have the lowest medians. Their intervals
  overlap with Qwen3-8B's, so no single fastest model is established.
- **Qwen3.5-9B.** The earlier explanation, "first-time CUDA-graph
  capture; subsequent recreates much faster", is not supported: the 10
  distinct launches from 2026-04-30 to 05-05 had median 143 s
  (140-153). Later
  launches, with other images and flags (2026-05-13 to 07-29), had
  median 91 s (n = 13).
- **The 600 s timeout.** Observed ready times are truncated at the
  timeout by construction, so they cannot show that the timeout is long
  enough. As data: the longest ready time was 278 s, reached twice
  (vllm-devai Qwen3.8-27B-MTP-devai-NVFP4 with empty engine caches, whose
  median was 271 s over n = 7, and vllm Qwen3.8-27B-MTP-NVFP4@98304 on
  2026-09-21 21:01Z). The 10 real launches that never became ready are
  listed above. Seven launches abandoned by router restarts had already
  run longer than 278 s (287, 328, 399, 461 and 491 s, plus two long
  bookkeeping gaps). The tail above 278 s was not observed; it is
  censored, not empty.

## Issues surfaced (model, serving and harness)

### 1. R1-Distill-Llama-8B emits raw byte-level BPE tokens
Its answers contain literal Llama-3 byte-level BPE markers, un-decoded:
U+0120 (space) and U+010A (newline). They appear in 50/50, 49/50 and
49/50 answer texts across the three retained HumanEval runs, and every
run scored 0/50. Most failures are syntax errors. In one run, 22/50
were NameErrors instead: U+0120 is a letter, so it fuses tokens into
identifiers. The mechanism is suspected to be an interaction between
`--reasoning-parser deepseek_r1` and the Llama-3 tokenizer in vLLM. The
Qwen-7B distill, with the same parser and a Qwen-2 tokenizer, shows no
markers in any of its 100 retained answers.

**Action**: file a vLLM issue with a reproducer. Until then, prefer
non-reasoning Llama-3 checkpoints such as Llama-3.1-8B-Instruct-NVFP4
(HumanEval 40/50 on 2026-05-05; the 0.72 quoted here earlier was the
2026-05-02 run under the v1 scorer, 36/50).

### 2. Tool-loop interruption on forced-mode models -- **FIXED**
The router's `tool_choice_pinning_required` HTTP 400 fired inside
inspect_ai's tool loop and aborted the task. inspect_ai resets
`state.tool_choice` to `"auto"` after a forced call, so the follow-up
turn tripped the router. `tasks/tools_use.py` now drives the loop
itself with `tool_loop_with_pin`: turn 1 pinned to
`ToolFunction(name=expect_tool)`, `execute_tools`, and for
result_followup only a second turn with `tool_choice="none"`.
`bench_runner.py` also passes `fail_on_error=False` for this task.

Verified on 2026-05-02, from the request bodies of the retained logs:
the three forced-mode models then scored 12/20 (R1-Distill-Llama-8B),
13/20 (R1-Distill-Qwen-7B) and 15/20 (Llama-3.1-8B). The earlier
"0.00" were not scores: those runs had aborted with errors.

**Tradeoffs.** In pinned mode multi_tool_pick hands the model the
answer. From this fix until 2026-07-20 *every* model was pinned, so all
2026-05-05 tools_use scores are pinned-protocol scores, while the
2026-05-02 scores of the auto-mode models came from the auto protocol.

### 3. Nemotron-Nano-9B-v2 leaks `</think>` markers
3 `</think>` matches across the 40 latency prompts (2026-05-05). The
model emits `<think>...</think>` inline (`cap=inline`), and in 2026-05
it had no `parsers:` block, so vLLM launched without
`--reasoning-parser`. NVIDIA's model card prescribes `/think` /
`/no_think` in the system or user message, a `<TOOLCALL>` format parsed
by the plugin `nemotron_toolcall_parser_no_streaming.py` via
`--tool-parser-plugin`, and `--trust-remote-code --mamba_ssm_cache_dtype
float32 --enable-auto-tool-choice --tool-call-parser nemotron_json`.
These were wired on 2026-07-20 (followup #4).

### 4. HumanEval scorer too strict for inline-reasoning models -- **FIXED**, with a new defect
The v1 `_clean_completion` matched only a fence that enclosed the
entire completion. A `<think>` preamble or surrounding prose made it
return the raw text, which then failed as a syntax error. The v2
cleaner (commit ad597ca, 2026-05-02) works in four steps:
1. strip `<think>...</think>` blocks;
2. return the body of the **last** fenced block;
3. otherwise, slice from `^def <entry_point>\(`;
4. otherwise, return the think-stripped text.

**Validation rerun, 2026-05-02** (paired, same 50 items):
- **Nemotron-Nano-9B-v2**: 0/50 -> 3/50 (b = 3, c = 0, exact McNemar
  p = 0.25). The change is not distinguishable from run-to-run
  variation. Only 4 of the 49 v1 syntax failures would even parse
  under v2.
  - The earlier conclusion "most failures are genuine coding weakness"
    is not supported: with the NVIDIA-prescribed parsers (2026-07-20) the
    model scored HumanEval 40/50 and HumanEval+ 44/50. The configuration
    also changed in between, so this is not a replicate, but the
    2026-05 score cannot be read as the model's coding ability.
- **R1-Distill-Llama-8B**: 0/50 -> 0/50. The BPE markers (Issue #1)
  dominate this failure.

**"No-op-or-improvement for already-passing models" is false.** The v2
fence regex strips the indentation of the first line of a fenced block
(defect D3). One item that passed under v1 in the 2026-05-02
Qwen3.5-9B run (HumanEval/40) no longer parses under v2. Later runs
contain 30 candidate false failures of the same kind, so the cached
scores of models that emit fenced function bodies can be biased
downwards. That includes the 2026-05-05 Qwen3.5-9B HumanEval, 39/50,
which could be up to 44/50 with a corrected extractor (unverified).

### 5. TPS undercounted for reasoning parsers (historical)
Superseded by "TPS counting fix" above. The "5-10x too low" written
here earlier disagreed with that section's 2.4-94x; both figures come
from the unretained first pass.

## KV-pressure observations

Peak VRAM relative to the card's 23.89 GiB (24,467 MiB; sold as 24
GB), single values from the 2026-05-05 sweep (GiB, device-wide
`nvidia-smi` maximum, sampled at 1 Hz):

| Model | Peak (GiB) | Percent of 23.89 GiB |
|---|---:|---:|
| Llama-3.1-8B-Instruct-NVFP4 @131K | 22.65 | 94.8 % |
| Qwen3-8B-NVFP4 @131K | 22.53 | 94.3 % |
| Nemotron-3-Nano-30B-A3B-NVFP4 @131K | 22.45 | 94.0 % |
| gpt-oss-20b @262K | 22.37 | 93.6 % |
| Qwen3-14B-NVFP4 @65K | 22.32 | 93.4 % |
| Nemotron-Nano-9B-v2-NVFP4 @65K | 22.28 | 93.3 % |
| DeepSeek-R1-Distill-Qwen-7B @65K | 21.90 | 91.7 % |
| DeepSeek-R1-Distill-Llama-8B @32K | 21.69 | 90.8 % |
| Qwen3.5-9B-NVFP4 @131K | 21.54 | 90.2 % |

**What these numbers are.**
- vLLM preallocates its memory pool (`--gpu-memory-utilization`) at
  launch, so peak VRAM mostly reflects that configured pool, not KV
  demand during the run (see [backends.md](backends.md), "What these
  fields are actually worth").
- The earlier labels "tight" and "comfortable", and the "0.95 x 24 =
  22.8 GB threshold where KV paging starts to bite", have no data
  behind them and are withdrawn.
- **No preemptions.** The end-of-run `vllm:num_preemptions_total` was
  0.0 for all nine models. This counter is cumulative for the
  container's life, so it covers the run.
- **Qwen3-14B-NVFP4.** The probe never offered a 128K tier for this
  model (a probe fact).
- **The 7-9 B context ceiling was wrong.** The earlier sentence
  "128-131K ... the practical ceiling for any 7-9 B NVFP4 model on 24
  GB" is contradicted by later rows: Qwen3.5-9B-NVFP4 and
  Ornith-1.0-9B-NVFP4 were benched on vLLM at 262144, with peaks of
  21.31 and 21.41 GiB.

KV-cache *quantization* (fp8 on vLLM, q8_0 on Ollama, and its measured
quality effects) is covered in [backends.md](backends.md) "Per-tier
KV-cache dtype".

## Picker-tier recommendations

The filter expressions encoded in the picker (`scripts/model-picker.py`,
constants `_PRODUCTION_AGENTIC_*`) are applied to point estimates from a
single run. The "matches" comments give the 2026-05-05 outcome together
with its robustness.

```
PRODUCTION_AGENTIC = (
    backend == "vllm"
    AND quantization in {"NVFP4", "MXFP4"}
    AND tools_use_score >= 0.9
    AND humaneval >= 0.7          # plain HumanEval (humaneval_subset_*),
                                  # NOT HumanEval+ (humaneval_plus_subset_*)
    AND gsm8k >= 0.9
    AND leak_rate == 0
    AND peak_vram_gb < 23
)
# matches (2026-05-05 point estimates): Nemotron-3-Nano-30B-A3B-NVFP4,
#   Qwen3-14B-NVFP4, Qwen3-8B-NVFP4, Qwen3.5-9B-NVFP4.
# Robustness: the tools_use 95% CI contains 0.9 for all four and for
#   gpt-oss-20b (17/20, CI 0.62-0.97); Qwen3.5-9B's HumanEval CI
#   (0.64-0.89) contains 0.7. Membership is not determined by the data.
# gpt-oss-20b: 18/20 on 2026-05-02 (tool_choice=auto) vs 17/20 on
#   2026-05-05 (pinned) -- different protocols, not distinguishable
#   (b=2, c=3, p=1.0); "slipped" is not supported.

CODING_SPECIALIST = max(humaneval) where PRODUCTION_AGENTIC
# 2026-05-05: Nemotron-3-Nano 49/50 has the highest point estimate among
#   the four; gpt-oss-20b (not AGENTIC) also 49/50. Not distinguishable
#   from Qwen3-14B 47/50 (p=0.63) or Qwen3-8B 44/50 (p=0.13).

LATENCY_SENSITIVE = (
    PRODUCTION_AGENTIC
    AND tps_sustained_p50 >= 50
)
# 2026-05-05: all four (55-144 tok/s, one run each). Qwen3.5-9B's 55.3
#   is 10 % above the cut-off: about 8 within-day single-run CVs (1.3 %)
#   and about 5 of its own across-month CVs (2.1 %; three runs,
#   55.27-57.55 tok/s).

THROUGHPUT_KING = max(tps_sustained_p50)
# 2026-05-05 (stale): Nemotron-3-Nano 143.8 vs gpt-oss-20b 139.2, a
#   3 % gap between two single-run medians -- not testable.
# Current cache: Nemotron-3-Nano's row reads 40.25 tok/s (2026-07-17,
#   --enforce-eager, ctx 163840); the maximum over all current rows is
#   174.75 (qwen3.6:35b-a3b-mtp-q4_K_M on Ollama, 131072), and 135.01
#   (gpt-oss-20b) among vLLM rows.

REASONING_ONLY = (
    backend == "vllm"
    AND gsm8k >= 0.85
    AND tps_sustained_p50 < 30
)
# 2026-05-05: no models (slowest qualifying: R1-Distill-Qwen-7B, 44.5).

AVOID = (
    leak_rate > 0
    OR has_known_decode_bug   # R1-Distill-Llama-8B
    OR (humaneval < 0.3 AND not REASONING_ONLY)
)
# 2026-05-05: R1-Distill-Llama-8B (BPE bug; HumanEval 0/50, CI 0-0.07),
#   Nemotron-Nano-9B-v2 (3 leak matches; no parsers configured then --
#   since 2026-07-20 its vLLM row reads leak 0/40, tools 20/20).
```

The picker reads `deploy/.bench-cache.json` directly and applies these
thresholds through `_is_production_agentic`. A tier label
therefore inherits the single-run, point-estimate nature of the numbers
([statistics-primer.md](statistics-primer.md) Sec. 10: a threshold
applied to a noisy score misclassifies near the cut-off).

## Followup work (ordered by impact)

- [x] ~~**Fix TPS counting**~~ **Done 2026-05-02**, with limits: the
  bench still counts characters/4 (open defect D4). The "validated on
  all 8 models" run was not retained; the drifts of 0.4-1.1 % quoted for
  it are within single-run variation (within-day CV about 1.3 %, 95% CI
  roughly 0.8-2.6 %). Nemotron-Nano-9B-v2
  was called "non-reasoning" in that validation, but it reasons inline.
  The Makefile also single-quotes `BENCH_REPO` / `BENCH_TASKS`.
1. ~~**Pin `tool_choice` per sample in `tools_use`**~~ **Done
   2026-05-02.** The forced-mode models Llama-3.1-8B-Instruct-NVFP4,
   DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Llama-8B
   produced 15/20, 13/20 and 12/20. Before the fix their runs aborted
   and produced no score. See Issue #2 for the protocol history.
2. ~~**Fix the HumanEval scorer for inline-reasoning models**~~ **Done
   2026-05-02**, but it introduced defect D3 (Issue #4). Nemotron-Nano
   0/50 -> 3/50 is not distinguishable at the 5% level (p = 0.25).
3. **Investigate the R1-Distill-Llama-8B BPE-decode bug** -- **in
   progress**. The reproducer is `scripts/repro/r1_distill_llama_bpe.py`,
   with a draft issue body in `scripts/repro/r1_distill_llama_bpe.md`.
   Its reported output ("9 U+0120 + 3 U+010A in a 33-char reply") was
   not retained; the retained HumanEval answers show the markers in
   49/50. **Remaining**: file the issue once the vLLM image digest to
   cite is settled.
4. ~~**Wire up Nemotron-Nano-9B-v2 per NVIDIA's official guidance**~~
   **Done 2026-07-20** (d09887c, 09af08b: reasoning parser with a
   per-model vLLM v0.25.1 pin; `nemotron_json` tool-parser plugin). The
   current vLLM row (ctx 131072) mixes configurations: HumanEval 40/50,
   HumanEval+ 44/50, tools_use 20/20 (auto) and leak 0/40 were measured
   on 2026-07-20 after the change; GSM8K 98/100 (07-17), MMLU-Pro 42/60
   and GPQA 39/60 (07-19) before it. Single runs under open defects
   D1-D3; no time-outs in this row.
5. ~~**Add a KV-pressure column to `make bench-report`**~~ **Done**
   (`KV %` = `peak_vram_gb / GPU_MEMORY_GB`). The 95 % threshold it
   footnotes is withdrawn ("KV-pressure observations"); the
   "Qwen3.5-9B-NVFP4 at 95.7 %" once quoted here is not retained and
   is inconsistent with the 21.54 GiB measured on 2026-05-05 (89.8 % of
   24, 90.2 % of the card's 23.89 GiB).
6. ~~**Add a long-context probe (one prompt at 80 % of ctx)**~~
   **Done** as the opt-in `longctx` task (`scripts/bench/bench_longctx.py`,
   `BENCH_TASKS=longctx,...`; knobs `--n-longctx-fraction`,
   `--n-longctx-max-tokens`): one completion at `fraction * ctx` input
   tokens, recording prefill TTFT, decode TPS, output tokens, finish
   reason, `vllm:kv_cache_usage_perc` and the preemption delta. The
   smoke test on Nemotron-3-Nano at 80 % of 131K (ttft 75.8 s, decode
   144.1 tok/s, 67 output tokens, peak 22.54 GiB) was one request and is
   not retained. Known limitation: `kv_cache_usage_perc` reads 0.0 at
   the end of the request.
7. ~~**Add a vLLM `/metrics` snapshot**~~ **Done.** An end-of-run,
   best-effort snapshot lands as `vllm_kv_cache_usage_perc` (gauge
   `vllm:kv_cache_usage_perc`) and `vllm_num_preemptions_total`;
   SGLang's metric names differ. It is not a maximum during the run:
   `kv_cache_usage_perc` falls once the queue drains.
8. ~~**Run `bench-sglang`**~~ **Done 2026-07-25..29** (gpt-oss-20b,
   Ornith-1.0-9B, Qwen3.5-9B, Nemotron-Nano-9B-v2). The comparison with
   vLLM is in [backends.md](backends.md) "vLLM vs SGLang -- measured,
   2026-07-29": confounded by context, KV dtype, run date, tool_choice
   protocol and time-outs, and no difference is distinguishable after a
   Holm correction over 18 tests.
9. ~~**Run `bench-ollama`**~~ **Partly done** from 2026-07-17 (gemma4,
   qwen3.5, qwen3.6 and qwen3.8 rows). Rows benched before 2026-09-19
   ran with 10 samples queued on Ollama's single slot; their time-outs
   (defect D2) depress slower models' scores (counts in the script
   output).
10. ~~**Wire the bench cache to a picker badge**~~ **Done.** The
    smoke-test list "4 models qualify (Qwen3-8B, Qwen3-14B, gpt-oss-20b,
    Nemotron-3-30B-A3B)" reflected the cache between 2026-05-05 09:23
    and 14:03 UTC (the 2026-05-02 rows plus the pre-fix Nemotron-3 run)
    and is stale.
11. ~~**Drop `--enforce-eager` on Nemotron-3-Nano by passing
    `--max-num-seqs 8`**~~ **Done 2026-05-05** (b730985). The OOM at
    model load was attributed to CUDA-graph capture buffers that scale
    with `max_num_seqs`; with 8 the model loaded with graphs enabled. The
    quoted numbers (TPS 42.87 -> 144.84, steady TTFT p50 70 -> 51 ms,
    peak 22.63 -> 22.43 GiB, cold start 65.6 s) come from runs that were
    not retained (the pre-fix 09:17 UTC run kept only its quality logs),
    and the post-fix figures differ from the sweep row (143.8 tok/s, 46.9
    ms, 22.45 GiB, 60.6 s). The retained speed evidence is one run per
    mode: 143.8 tok/s with CUDA graphs (sweep, ctx 131072) and 40.25 with
    `--enforce-eager` (current cache row, 2026-07-17, ctx 163840; the
    2026-07-17..19 launches log `enforce_eager: True`), confounded by
    context, `--max-num-seqs` and date. Quality before and after the fix:
    GSM8K 98 vs 99, HumanEval 43 vs 49 (raw p = 0.070), tools_use 20 vs
    20. That the "doesn't investigate deep enough" symptom seen
    in a parallel Claude Code session vanished after the speed-up is an
    anecdote; "throughput regressions can masquerade as quality
    regressions" is a hypothesis nothing here tests
    ([statistics-primer.md](statistics-primer.md) Sec. 7, "Explaining
    after the fact").
12. **Open harness defects D1-D4** ("Methodology"): pass sampling as
    `eval()` keywords; record time-outs as their own outcome and stop
    counting queueing against the limit; stop the fence regex consuming
    indentation; request `stream_options.include_usage`. Rows measured
    before a fix are not comparable with rows measured after it.
13. **Replication.** At least two runs per cell, or report every run,
    so run-to-run variation is measured rather than inferred from
    incidental re-runs.

## Reproducing the statistics

All scripts are stdlib-only Python 3, deterministic (fixed seeds, B =
10000) and read-only on their inputs; full commands and inputs are in
[scripts/stats/README.md](../scripts/stats/README.md). The shared
library `scripts/stats/statlib.py` is tested against published values by
`tests/python/test_stats_statlib.py`.

```
L=/var/cache/devai/bench/inspect-logs; O=~/.cache/devai/stats/bench; mkdir -p $O
python3 scripts/stats/bench_extract_logs.py $L $O/logs_extracted.json
python3 scripts/stats/bench_stats.py $O/logs_extracted.json $L deploy/.bench-cache.json $O
python3 scripts/stats/bench_scorer_scan.py $O/logs_extracted.json $L $O/scorer_artifact_scan.json   # D3, parse-only
python3 scripts/stats/bench_engine_facts.py /var/cache/devai/logs $O/engine_log_facts.json          # D1 samplers, KV dtype
python3 scripts/stats/bench_timeouts.py $L $O/timeouts.tsv                                          # D2
grep -E "^=== |ttft_first|score:|pass@1:|by_subcase|done in|/metrics" \
     /var/cache/devai/bench/bench-vllm-run-20260505T140328Z.log   # latency/TPS/VRAM/leak cells
```

`$O/results.md` holds the 2026-05-05 leaderboard with t and intervals;
all 36 aggregate and 24 per-task comparisons; the threshold-robustness
table and paired change comparisons; the incidental-replication table,
annotated with the configuration changes known between runs; the
vLLM-vs-SGLang tests; and the current cache in the standard layout with
time-out counts and ranges. The speed figures (cold-start medians,
decode ceilings, run-to-run TPS variation, chars/4 against engine
counts) come from `perf_coldstart.py`, `perf_ceiling.py`,
`perf_bench_cache.py` and `perf_engine_runs.py` in the same directory.

## Cross-references

- [statistics-primer.md](statistics-primer.md) -- how to read the
  intervals, tests and limitations used on this page.
- [router.md](router.md) -- request rewrite chain, including the
  `tool_choice_pinning_required` rule that the bench surfaced as Issue
  #2, and the benchmark harness reference.
- [backends.md](backends.md) -- backend lifecycle, parser plugins, the
  vLLM-vs-SGLang comparison, and "Per-tier KV-cache dtype".
- [nvfp4-coldstart.md](nvfp4-coldstart.md) -- NVFP4 cold-start phases;
  `ttft_ms_first` includes that timeline only when the first request
  triggered the launch.
- `deploy/.bench-cache.json` -- the current cache (not the 2026-05-05
  rows).
- `/var/cache/devai/bench/inspect-logs/*.eval` -- full per-sample
  inspect_ai logs (zip archives of JSON), viewable with
  `inspect view start --log-dir /var/cache/devai/bench/inspect-logs`.
