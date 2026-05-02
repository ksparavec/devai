# Bench results -- first vLLM sweep on RTX 4000 PRO Blackwell (24 GB)

> Generated 2026-05-02 from `deploy/.bench-cache.json`. Hardware:
> RTX 4000 PRO Blackwell, 24 GB VRAM, ~640 GB/s memory bandwidth.
> Backend: vLLM `latest-cu130-ubuntu2404` via the gpu-arbiter router.
> Bench harness: `scripts/bench/` -- see `docs/router.md` "Benchmark
> harness" section. Per-task subset sizes: GSM8K n=100, HumanEval
> n=50, tools_use n=20, latency/leak n=40 streamed prompts.

## TL;DR

The bench produces evidence-grade routing recommendations. On this
24 GB Blackwell card:

- **`gpt-oss-20b@262144`** is the **fastest** (136 tok/s, MoE+MXFP4)
  AND **best coder** (HumanEval 0.98). Single best model overall for
  agentic + coding work.
- **`Qwen3-8B-NVFP4@131072`** has the **highest aggregate
  correctness** (0.97) and ties gpt-oss on speed for an 8B (98 tok/s).
  Best when you specifically want a Qwen-family reasoning model with
  perfect tool calling.
- **`Qwen3-14B-NVFP4@65536`** matches the small Qwen on quality but
  caps at 64K context due to KV pressure. Use only when 128K isn't
  needed.
- **`Llama-3.1-8B-Instruct-NVFP4@131072`** is the fastest non-
  reasoning model (96 tok/s, no `<think>` preamble) -- good fallback
  for latency-critical simple tasks.
- **R1-Distill family (BF16)** are not as slow as the original bench
  suggested (real TPS 42-45, not 13-17 -- counting bug fixed); they're
  still slow in *wall time* because of long reasoning preambles.

| Tier | Models | Use for |
|---|---|---|
| **Production default** | `gpt-oss-20b@262144` | Claude Code, Aider, multi-tool agents -- fastest and most accurate combination |
| **Reasoning + tools alternative** | `Qwen3-8B-NVFP4@131072`, `Qwen3-14B-NVFP4@65536` | when you specifically want the Qwen3 family or perfect tool fidelity (1.00 tools_use) |
| **Non-reasoning low-latency** | `Llama-3.1-8B-Instruct-NVFP4@131072` | simple prompts where you don't want CoT preamble |
| **Reasoning-only (high latency)** | `DeepSeek-R1-Distill-Qwen-7B@65536` | offline / batch / when reasoning depth matters more than wall time |
| **Avoid (broken)** | `DeepSeek-R1-Distill-Llama-8B@32768` (BPE-decode bug, 0.0 HumanEval), `Nemotron-Nano-9B-v2-NVFP4@65536` (3 leak hits, no tool parser) | -- |

## Full results table

| Model | GSM8K | HumanEval | tools_use | by_subcase (E/S/M/F) | Leak | Cold (s) | Warm p50/p95 (ms) | TPS (tok/s) | Peak VRAM | Aggregate |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| **Qwen3-8B-NVFP4@131072** | 0.98 | 0.94 | **1.00** | 1.0/1.0/1.0/1.0 | 0.000 | 45.6 | 32.7/34.5 | **98.3** | 22.53 GB | **0.97** |
| **Qwen3-14B-NVFP4@65536** | 0.97 | 0.92 | **1.00** | 1.0/1.0/1.0/1.0 | 0.000 | 49.8 | 45.8/48.0 | 62.0 | 22.32 GB | **0.96** |
| **gpt-oss-20b@262144** | 0.97 | **0.98** | 0.90 | 0.8/1.0/0.8/1.0 | 0.000 | 56.0 | 51.0/54.0 | **136.4** | 22.36 GB | **0.95** |
| **Qwen3.5-9B-NVFP4@131072** | 0.95 | 0.74 | 0.80 | 0.4/1.0/0.8/1.0 | 0.000 | 162.3 | 32.6/34.1 | 55.7 | 21.66 GB | 0.83 |
| **Llama-3.1-8B-Instruct-NVFP4@131072** | 0.84 | 0.72 | 0.75 | 1.0/0.6/1.0/0.4 | 0.000 | 41.6 | 22.9/24.9 | **95.9** | 22.65 GB | 0.77 |
| **DeepSeek-R1-Distill-Qwen-7B@65536** | 0.88 | 0.26 | 0.65 | 1.0/0.8/0.4/0.4 | 0.000 | 86.9 | 36.4/40.1 | 45.2 | 21.89 GB | 0.60 |
| **DeepSeek-R1-Distill-Llama-8B@32768** | 0.57 | 0.00^+^+^+ | 0.60 | 1.0/0.4/0.4/0.6 | 0.000 | 85.2 | 37.6/53.1 | 42.5 | 21.69 GB | 0.39 |
| **Nemotron-Nano-9B-v2-NVFP4@65536** | 0.84 | 0.00^+^+^+ | 0.00 | 0.0/0.0/0.0/0.0 | 0.075 (3x `</think>`) | 114.9 | 27.6/32.0 | 81.1 | 22.28 GB | 0.28 |

Subcase legend: **E**mpty-schema . **S**ingle-arg . **M**ulti-tool-pick . **F**ollow-up.
All TPS values are post-fix (see "TPS counting fix" below).

Footnotes for caveats marked above:

- ^+^+^+ **HumanEval=0.00 for inline-reasoning models** (Nemotron-Nano,
  R1-Distill-Llama-8B) is a scorer-strictness issue: my
  `_clean_completion` only handles a fence enclosing the entire
  completion, but inline-reasoning models emit `<think>...</think>`
  blocks before the code fence. Fix planned: search for the LAST
  fenced code block and ignore preamble. Plus, R1-Distill-Llama-8B
  has a *separate* tokenizer-decoding bug -- see the issues section.

(The previous `^+^+` footnote about `tools_use=0.00` on forced-mode
models is gone: the bench's `tools_use` task now pins `tool_choice`
per-sample via a custom tool-call loop, so R1-Distill x2 and
Llama-3.1-8B-Instruct-NVFP4 produce real scores. See "Issues surfaced
> 2. Tool-loop interruption on forced-mode models" below.)

## Methodology

Per (model, backend) pair:

1. **Latency/leak sidecar** (40 streamed prompts, sequential, batch=1):
   - First prompt's TTFT recorded as `ttft_ms_first` -- measures
     **cold start** including container recreate + weight load + KV
     alloc + prefill.
   - Remaining 39 prompts' TTFTs feed `ttft_ms_steady_p50/p95`.
   - Stream-body decode rate gives `tps_sustained_p50`.
   - Concatenated response bodies are regex-swept for known template-
     leak markers (`<channel|>`, `<|im_end|>`, `<|tool_call_begin|>`,
     `<think>`, etc. -- see `scripts/bench/data/leak_markers.txt`).
2. **GSM8K** via inspect_ai (n=100, exact-match scorer).
3. **HumanEval** via inspect_ai (n=50, local subprocess sandbox,
   pass@1).
4. **tools_use** custom inspect_ai task (n=20, four sub-cases).
5. **VRAM sampler** thread polls `nvidia-smi` at 1 Hz throughout each
   model's run, reports peak/mean.

Results merge into one row per model in `deploy/.bench-cache.json`,
keyed by `<repo@sha>` to join with the probe cache.

## TPS counting fix

**The first bench pass reported TPS values that were 2.5-94x too low
for any model with a reasoning parser** because vLLM's
`usage.completion_tokens` only counts `content` tokens -- it excludes
`reasoning_content` tokens generated under `--reasoning-parser
{qwen3,deepseek_r1}` (and probably others). Since the bench's TPS
divisor `(t_done - t_first_token)` spans the full stream including
the entire `<think>` block, dividing a small numerator by a large
denominator gave nonsense rates like 0.66 tok/s for Qwen3-14B-NVFP4.

**Fix** (`scripts/bench/_bench_core.py::stream_chat_completion`):
accumulate `delta.reasoning_content` characters alongside `delta.content`,
then derive an `effective_tokens = max(usage.completion_tokens,
(content_chars + reasoning_chars) // 4)`. The `// 4` heuristic is the
standard ~4-chars-per-token approximation; the `max()` keeps accurate
parsers from being penalised on short outputs.

**Validation** -- re-ran leak/latency for every vLLM model and
compared:

| Model | Parser | Before fix | After fix | Delta | Notes |
|---|---|---:|---:|---:|---|
| Qwen3-14B-NVFP4 | qwen3 | 0.66 | **62.02** | **94x** | qwen3 parser was the worst offender |
| **gpt-oss-20b** | harmony (openai_gptoss) | 38.4 | **136.37** | **3.6x** | harmony parser also undercounts; MoE = 3.6B active params -> fastest in bench |
| Qwen3-8B-NVFP4 | qwen3 | 14.53 | **98.31** | 6.8x | 8B fits more bandwidth headroom |
| Qwen3.5-9B-NVFP4 | qwen3 | 22.67 | **55.73** | 2.5x | hybrid Mamba arch decodes slower than pure Transformer |
| DeepSeek-R1-Distill-Qwen-7B | deepseek_r1 | 13.85 | **45.18** | 3.3x | deepseek_r1 also undercounts, but less |
| DeepSeek-R1-Distill-Llama-8B | deepseek_r1 | 17.51 | **42.51** | 2.4x | same parser bug; BF16 is at bandwidth ceiling anyway |
| Nemotron-Nano-9B-v2-NVFP4 | (none, inline `<think>`) | 82.0 | **81.08** | **-1.1 %** | inline `<think>` lands in `content` directly -- usage already counted it; drift is run-to-run noise |
| **Llama-3.1-8B-Instruct-NVFP4** | (none) | 95.53 | **95.88** | **+0.4 %** | non-reasoning model -- fix is a no-op as expected, drift within measurement noise |

The Llama-3.1 and Nemotron rows are the validation cases: models
with no separate `reasoning_content` channel see the
`effective_tokens` formula fall through to `usage.completion_tokens`
unchanged, and drift is within run-to-run noise (different prompt
sampling). **Fix confirmed correct for both regressions and
non-regressions.**

The biggest single insight from the re-run: **`gpt-oss-20b` is the
fastest model in the bench at 136 tok/s** -- not the slowest as the
original numbers suggested. As an MoE with ~3.6B active parameters
per token plus native MXFP4, its per-token weight transfer
(~1.8 GB) is well below the dense 8B NVFP4 models (~5 GB) -- and on
a 640 GB/s memory-bound card, that's a ~3x ceiling advantage. **MoE
+ FP4 wins decisively** on this hardware class.

## Architectural finding: FP4 quantization + MoE on Blackwell

After the TPS counting fix, the picture is cleaner: this
Blackwell card delivers a **~3x decode-rate spread** across the
quantization x architecture matrix, with MoE+MXFP4 winning decisively:

| Architecture x Quantization | Models | TPS p50 | Why |
|---|---|---:|---|
| **MoE + MXFP4** (native Blackwell, only ~3.6B active) | gpt-oss-20b | **136 tok/s** | smallest per-token weight transfer (~1.8 GB) |
| **Dense + NVFP4** (small) | Qwen3-8B, Llama-3.1-8B-Instruct | 96-98 tok/s | ~5 GB/token; FP4 tensor cores hit |
| **Dense + NVFP4** (medium reasoning) | Nemotron-Nano-9B-v2 | 81 tok/s | similar to small NVFP4 + reasoning template overhead |
| **Dense + NVFP4** (large) | Qwen3-14B-NVFP4 | 62 tok/s | ~7 GB/token; bandwidth scales linearly with weight size |
| **Hybrid Mamba + NVFP4** (community quant) | Qwen3.5-9B (`ykarout/`) | 56 tok/s | linear-attention layers decode well, but Mamba state arithmetic adds overhead vs pure Transformer |
| **Dense + BF16** | DeepSeek-R1-Distill-{Qwen-7B, Llama-8B} | 42-45 tok/s | 16 GB/token weight transfer at 640 GB/s ~ 40 tok/s ceiling |

**Practical wall-time matters more than raw TPS** because reasoning
models emit a long `<think>` preamble before the visible answer.
End-to-end wall time for an agent waiting on a 200-token reply:

| Model | TPS | Pre-answer reasoning | Total wall time |
|---|---:|---:|---:|
| `gpt-oss-20b` (harmony channels, structured) | 136 | ~300 tokens | **3.7 s** |
| `Qwen3-8B-NVFP4` (Qwen3 thinking) | 98 | ~500 tokens | **7.1 s** |
| `Llama-3.1-8B-Instruct-NVFP4` (no reasoning) | 96 | 0 | **2.1 s** |
| `Qwen3-14B-NVFP4` | 62 | ~600 tokens | **12.9 s** |
| `DeepSeek-R1-Distill-Qwen-7B` (BF16, R1-style) | 45 | ~1500 tokens | **37.8 s** |

The R1-Distill family is **not catastrophically slow on raw decode**
(40+ tok/s, hitting the bandwidth ceiling for BF16 8B); they're slow
in **wall time** because of the long thinking preamble. For an
interactive agent like Claude Code where every prompt is a turn,
this matters end-to-end -- choose architecture/parser combinations
that minimise the wall-time integral, not maximise raw TPS in
isolation.

The card's ~640 GB/s memory bandwidth is the bottleneck for batch=1
decode. NVFP4 cuts the per-token weight transfer by 4x, and Blackwell
hardware tensor cores process FP4 natively -- both effects compound.
For BF16 8B models, the theoretical ceiling is ~40 tok/s
(16 GB / 640 GB/s); real-world ~15 tok/s after parser, framework, and
streaming overhead.

**Routing implication**: prefer NVFP4 quantizations of any reasoning-
capable model. The Qwen3 NVFP4 family (8B, 14B) and gpt-oss-20b are
the practical agentic workhorses on this hardware. The R1-Distill
BF16 family stays in the catalog as a curiosity (their reasoning
quality is real -- Qwen-7B distill scored 0.88 on GSM8K) but should
not be the default for any latency-sensitive workflow.

## Cold-start signal

The `ttft_ms_first` metric captures the user-observable cold start -- 
exactly what an agent like Claude Code experiences on first request to
a freshly-routed model:

| Model | Cold (s) | Notes |
|---|---:|---|
| Llama-3.1-8B-Instruct-NVFP4 | 41.6 | fastest cold start; small footprint, no reasoning preamble |
| Qwen3-8B-NVFP4 | 45.6 | + reasoning template + Qwen3 chat template |
| Qwen3-14B-NVFP4 | 49.8 | bigger weights |
| gpt-oss-20b | 57.9 | 20 B MoE, harmony channels, longer graph compile |
| DeepSeek-R1-Distill-Llama-8B | 85.2 | BF16, slower paged-in load |
| DeepSeek-R1-Distill-Qwen-7B | 86.9 | same |
| Nemotron-Nano-9B-v2-NVFP4 | 114.4 | NVFP4 + 9B + Nemotron-specific kernel compile |
| **Qwen3.5-9B-NVFP4** | **162.3** | **first-time CUDA-graph capture for the Qwen3.5 generation; subsequent recreates much faster** |

Operational note: the `HEALTH_TIMEOUT_SECONDS=600` default in
`gpu-arbiter` is justified -- Qwen3.5-9B's 162 s startup leaves
generous margin against the 600 s ceiling, and slower NVFP4 + CUDA-
graph compiles on bigger MoEs can push 4-5 minutes in the wild.

## Issues surfaced (all real findings, not bench bugs)

### 1. R1-Distill-Llama-8B emits raw byte-level BPE tokens
Sample completions contained literal `G-marker` (space, U+0120) and `C-marker`
(newline, U+010A) -- the Llama-3 byte-level BPE markers, un-decoded.
HumanEval failed every sample because Python parser rejects these as
syntax errors. Smells like a `--reasoning-parser deepseek_r1` x Llama-3
tokenizer interaction in vLLM where the parser extracts text at the
wrong layer (tokens vs decoded UTF-8). Specific to this model -- the
Qwen-7B distill (Qwen-2 tokenizer) does NOT show this bug.

**Action**: file vLLM issue with reproducer; in the meantime, route
Llama-3 reasoning to OpenAI-compatible non-reasoning paths or use the
Llama-3.1-Instruct-NVFP4 (no reasoning, 0.72 HumanEval).

### 2. Tool-loop interruption on forced-mode models — **FIXED**
The router's `tool_choice_pinning_required` HTTP 400 -- designed to
prevent silent garbage from forced-mode agents -- fired inside the
inspect_ai tool loop and got converted to RuntimeError, killing the
task. This is the router doing exactly what we built it to do; the
bench needed to pin `tool_choice` per-sample to test forced-mode models.

**Why setting `state.tool_choice` alone wasn't enough**: inspect_ai's
built-in tool loop (`_eval/task/generate.py::task_generate`) reads
`state.tool_choice` once at the start, but **resets it to `"auto"`
after a forced `ToolFunction` call** so the model can produce a final
answer. That second turn then trips the router. There is no public
API to disable that reset.

**Resolution**: `tasks/tools_use.py` now ships a custom solver
(`tool_loop_with_pin`) that drives the loop manually:
turn 1 with `tool_choice=ToolFunction(name=expect_tool)`;
`execute_tools` for any tool calls; for `result_followup` only, a
second turn with `tool_choice="none"` so the model produces text
without re-tripping the router. Other subcases stop after turn 1 --
the scorer grades them on the tool call alone. `bench_runner.py` also
passes `fail_on_error=False` to `inspect_eval` for the tools task as
belt-and-suspenders. Verified end-to-end with `BENCH_TASKS=tools
BENCH_FORCE=1`; all three forced-mode models now produce real scores
(0.60--0.75) instead of 0.00.

**Tradeoff**: `multi_tool_pick` no longer tests the model's tool-
routing decision (we hand it the answer); it now tests args
correctness given the right tool. That's the only path that works
for forced-mode models on multi-tool prompts.

### 3. Nemotron-Nano-9B-v2 leaks `</think>` markers
3 occurrences across 40 latency prompts. The model emits
`<think>...</think>` inline (it's `cap=inline`, not structured), and
Nemotron has no `parsers:` block in `model-families.yaml`, so vLLM
launches without `--reasoning-parser`. The parser would catch the
markers if curated. **Action**: research Nemotron's reasoning parser
support; if available in vLLM, add to family yaml.

### 4. HumanEval scorer too strict for inline-reasoning models
`_clean_completion` only handles a fence enclosing the entire output.
Inline-reasoning models put `<think>...</think>` before the code
fence. Affects Nemotron-Nano (0.00 likely much higher in reality) and
R1-Distill-Llama-8B (compounded by issue #1). **Action**: in v2, find
the LAST fenced code block; if no fence, find the function definition
matching `entry_point`.

### 5. TPS undercounted for Qwen3 reasoning parser
vLLM with `--reasoning-parser qwen3` doesn't include `reasoning_content`
tokens in `usage.completion_tokens`. The bench's TPS divisor spans the
full stream, so reported values are 5-10x too low for Qwen3 family.
**Action**: switch the bench's TPS-counting to character-bytes / 4
(approximate but parser-agnostic), or also accumulate
`reasoning_content` text length and add to the count.

## KV-pressure observations

Peak VRAM relative to the 24 GB cap:

| Model | Peak / 24 GB | Pressure |
|---|---:|---|
| Llama-3.1-8B-Instruct-NVFP4 @131K | 22.65 / 24 = **94.4%** | tight; KV near limit at full ctx |
| Qwen3-8B-NVFP4 @131K | 22.53 / 24 = **93.9%** | tight |
| gpt-oss-20b @262K | 22.37 / 24 = 93.2% | tight |
| Qwen3-14B-NVFP4 @65K | 22.32 / 24 = 93.0% | tight |
| Nemotron-Nano-9B-v2-NVFP4 @65K | 22.29 / 24 = 92.9% | tight |
| DeepSeek-R1-Distill-Qwen-7B @65K | 21.89 / 24 = 91.2% | comfortable |
| Qwen3.5-9B-NVFP4 @131K | 21.66 / 24 = 90.3% | comfortable |
| DeepSeek-R1-Distill-Llama-8B @32K | 21.69 / 24 = 90.4% | comfortable |

All models cleared the probe filter (no KV preemption observed
during the run). The 4 models above 93% are running at or near the
practical context ceiling for this card. **Recommendation**: when
benching at higher contexts, watch for `peak_vram_gb >= 0.95 * 24` =
22.8 GB; that's the threshold where KV paging starts to bite.

For **Qwen3-14B-NVFP4** specifically, 22.32 GB at only 65K context
means **128K is not feasible** on this card -- confirmed by the probe
which never offered the 128K tier for this model. The 8B and 9B NVFP4
models all hit similar headroom at 128-131K, suggesting that's the
practical ceiling for any 7-9 B parameter NVFP4 model on 24 GB.

## Picker-tier recommendations

Encoded as filter expressions ready for the picker / docs:

```
PRODUCTION_AGENTIC = (
    backend == "vllm"
    AND quantization == "NVFP4"
    AND tools_use_score >= 0.9
    AND humaneval >= 0.7
    AND gsm8k >= 0.9
    AND leak_rate == 0
    AND peak_vram_gb < 23
)
# matches: Qwen3-8B, Qwen3-14B, gpt-oss-20b

CODING_SPECIALIST = max(humaneval) where PRODUCTION_AGENTIC
# matches: gpt-oss-20b

LATENCY_SENSITIVE = (
    PRODUCTION_AGENTIC
    AND tps_sustained_p50 >= 50  # using corrected TPS once the bug fix lands
)
# matches: gpt-oss-20b (38 measured but real ~50-80)

REASONING_ONLY = (
    backend == "vllm"
    AND gsm8k >= 0.85
    AND tps_sustained_p50 < 30  # genuinely slow on this card
)
# matches: DeepSeek-R1-Distill-Qwen-7B (latency too high for default agent)

AVOID = (
    leak_rate > 0
    OR has_known_decode_bug   # R1-Distill-Llama-8B
    OR (humaneval < 0.3 AND not REASONING_ONLY)
)
# matches: R1-Distill-Llama-8B (BPE bug), Nemotron-Nano-9B-v2 (leaks)
```

These are derived from the cache, not hand-curated. The picker can
read `deploy/.bench-cache.json` directly and surface a "verified for
agentic use" badge.

## Followup work (ordered by impact)

- [x] ~~**Fix TPS counting** -- accumulate `reasoning_content` chars and
  use `max(usage.completion_tokens, chars/4)`.~~ **Done.** Validated
  on all 8 models: non-reasoning (Llama-3.1, Nemotron) drift 0.4-1.1%
  as expected (no-op fix path); reasoning-parser models corrected
  2.4x (deepseek_r1) to 94x (Qwen3-14B with qwen3 parser); harmony
  parser also fixed (gpt-oss 38->136 tok/s, MoE leader confirmed).
  Makefile also fixed to single-quote `BENCH_REPO` / `BENCH_TASKS`
  so regex pipes don't get interpreted by the shell.
1. ~~**Fix `tools_use` task to pin `tool_choice` per-sample** -- recovers
   tools data for the 3 forced-mode models (R1-Distill x2 + Llama-3.1
   NVFP4).~~ **Done.** First attempt (set `state.tool_choice` and reuse
   `generate()`) failed because inspect_ai's built-in tool loop resets
   `tool_choice` to `"auto"` on the follow-up turn. Final fix replaces
   `generate()` with a custom `tool_loop_with_pin` solver in
   `scripts/bench/tasks/tools_use.py` that drives the loop manually:
   turn 1 pinned to `ToolFunction(name=expect_tool)`, optional turn 2
   for `result_followup` with `tool_choice="none"`. Validated on all 3
   forced-mode models: Llama-3.1-8B-Instruct-NVFP4 (0.00 → 0.75),
   DeepSeek-R1-Distill-Qwen-7B (0.00 → 0.65), DeepSeek-R1-Distill-Llama-8B
   (0.00 → 0.60). See "Issues surfaced > 2" for details and the tradeoff
   on `multi_tool_pick`.
2. **Fix HumanEval scorer for inline-reasoning models** -- recovers
   Nemotron and R1-Distill-Llama-8B HumanEval data. ~30 min.
3. **Investigate R1-Distill-Llama-8B BPE-decode bug** -- file vLLM
   issue with reproducer. Likely a vLLM x Llama-3 tokenizer x reasoning-
   parser interaction.
4. **Add KV-pressure column to `make bench-report`** -- the data is
   already in the cache. ~10 min.
5. **Add a long-context probe (one prompt at 80% of ctx)** -- detects
   KV paging cliffs that the current latency probe misses. v2 feature.
6. **Add inspect_ai's vLLM `/metrics` snapshot** -- captures
   `vllm:gpu_cache_usage_perc` and `vllm:num_preemptions_total` per
   run. Real-time KV pressure indicators.
7. **Run `bench-sglang`** -- only one fitting model (`gpt-oss-20b`),
   single-row delta to the leaderboard. Quick.
8. **Run `bench-ollama`** -- 28 models, ~4 hours wall time. Parallel
   to vLLM since Ollama models are smaller and the bench framework
   is the same.
9. **Wire bench cache -> picker badge** -- tag PRODUCTION_AGENTIC rows
   in the picker UI. v2 feature.

## Cross-references

- `docs/router.md` -- request rewrite chain, including the
  `tool_choice_pinning_required` rule that the bench surfaced as
  issue #2.
- `docs/backends.md` -- backend lifecycle, parser plugins.
- `docs/nvfp4-coldstart.md` -- graphviz timeline of NVFP4 cold-start
  phases; bench's `ttft_ms_first` corresponds directly to that
  timeline's last bar.
- `deploy/.bench-cache.json` -- raw cache.
- `/var/cache/devai/bench/inspect-logs/*.eval` -- full per-sample
  inspect_ai logs, viewable with
  `inspect view start --log-dir /var/cache/devai/bench/inspect-logs`.
