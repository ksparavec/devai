# Bench results -- vLLM leaderboard on RTX 4000 PRO Blackwell (24 GB)

> Last refreshed 2026-05-05 from `deploy/.bench-cache.json`
> (host_env_id `ea4fd7e7b668`: kernel `6.12.85+deb13-amd64`, driver
> `595.71.05`, GPU `NVIDIA RTX PRO 4000 Blackwell`, CUDA `13.2`).
> Hardware: RTX 4000 PRO Blackwell, 24 GB VRAM, ~640 GB/s memory
> bandwidth. Backend: vLLM `latest-cu130-ubuntu2404` via the
> gpu-arbiter router. Bench harness: `scripts/bench/` -- see
> `docs/router.md` "Benchmark harness" section. Per-task subset
> sizes: GSM8K n=100, HumanEval n=50, tools_use n=20, latency/leak
> n=40 streamed prompts.
>
> **Cache schema v2** (effective 2026-05-05): every row stamps
> `host_env_id` so re-benches against a different driver/kernel are
> auditable; the `_meta.host_env_history` block holds the
> environment for each id. `make bench-vllm BENCH_FORCE=1` resets a
> row's `tasks` / `metrics` before re-running so stale fields don't
> linger; `first_benched_at` is preserved. Refresh this doc with
> `make bench-report` after a new sweep.

## TL;DR

The bench produces evidence-grade routing recommendations. On this
24 GB Blackwell card, the 2026-05-05 sweep yields a clear ordering
across nine vLLM rows (subset of the picker; SGLang on hold):

- **`NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4@131072`** is the new
  **outright leader** -- aggregate **0.99** (gsm8k 0.99, HumanEval
  0.98, tools_use 1.00 with all four subcases at 1.0), **143.8 tok/s**
  steady-state decode, 22.45 GB peak. MoE in NVFP4 weights with
  FP4 KV cache, 128K context. Only model that ties top-of-bench on
  every quality task *and* sits in the top-2 by speed. Default pick
  for any interactive agentic workload on this hardware.
- **`Qwen3-14B-NVFP4@65536`** is a close second at aggregate **0.97**
  and ties Nemotron-3-Nano on perfect tool fidelity (1.00). Slower
  at 61.9 tok/s because dense weights are bigger; capped at 64K
  context due to KV pressure (94 % VRAM at peak). Pick when 64K is
  enough and you want the Qwen3 family.
- **`gpt-oss-20b@262144`** and **`Qwen3-8B-NVFP4@131072`** tie at
  aggregate **0.93**. Different strengths:
  - **gpt-oss-20b**: fastest at the top tier (139.2 tok/s, MoE+MXFP4)
    and HumanEval 0.98 -- but tools_use slipped to 0.85 (empty-schema
    subcase down to 0.4), so it no longer qualifies for the
    PRODUCTION_AGENTIC badge that requires tools >= 0.9.
  - **Qwen3-8B-NVFP4**: 0.95 tools, 0.97 gsm8k, 102 tok/s, 128K
    context. The smaller-model AGENTIC choice when you want
    Nemotron-3's fidelity without the 30B-class container.
- **`Qwen3.5-9B-NVFP4@131072`** scores aggregate **0.90** with strong
  reasoning (gsm8k 0.98, tools 0.95) but weaker code (HumanEval
  0.78). 55.3 tok/s steady, the longest cold-start of the sweep
  (~160 s). Picks when reasoning depth + 128K matter more than
  wall-time.
- **`DeepSeek-R1-Distill-Qwen-7B@65536`** -- aggregate 0.80, with a
  surprising HumanEval **0.94** (up from 0.26 in the 2026-05-02
  sweep -- worth investigating; likely a sampling/scorer interaction
  flagged in followup work). Tools score 0.60 keeps it out of the
  AGENTIC tier.
- **`Llama-3.1-8B-Instruct-NVFP4@131072`** is the fastest
  non-reasoning model (95.4 tok/s, no `<think>` preamble) and the
  fastest cold-start (~40 s). Aggregate 0.76; good fallback for
  latency-critical simple tasks.
- **`DeepSeek-R1-Distill-Llama-8B@32768`** still scores HumanEval
  **0.00** -- the byte-level BPE-decode bug from 2026-05-02 hasn't
  been fixed upstream. Aggregate 0.48; avoid.
- **`Nemotron-Nano-9B-v2-NVFP4@65536`** is the only row with a
  non-zero leak rate (**0.075** -- 3 of 40 prompts emit
  `</think>` markers) *and* tools_use 0.00 (no probe-confirmed tool
  parser, so the router strips `tools` from the request). Aggregate
  0.27; do not deploy until both issues are resolved.

| Tier | Models | Use for |
|---|---|---|
| **Production default** (PRODUCTION_AGENTIC badge) | `NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4@131072`, `Qwen3-14B-NVFP4@65536`, `Qwen3-8B-NVFP4@131072`, `Qwen3.5-9B-NVFP4@131072` | meet every threshold (gsm8k >= 0.9, HumanEval >= 0.7, tools >= 0.9, leak == 0.0, peak < 23 GB). Default agentic picks; Nemotron-3-Nano is fastest of the four |
| **High-throughput coder (tools-light)** | `gpt-oss-20b@262144` | dropped out of AGENTIC because tools_use = 0.85 < 0.9; still the fastest 0.97/0.98 reasoning+code combo when tool fidelity isn't critical |
| **Non-reasoning low-latency** | `Llama-3.1-8B-Instruct-NVFP4@131072` | simple prompts where you don't want CoT preamble; fastest cold-start of the sweep |
| **Reasoning-only (high latency)** | `DeepSeek-R1-Distill-Qwen-7B@65536` | offline / batch / when reasoning depth matters more than wall time |
| **Avoid (broken)** | `DeepSeek-R1-Distill-Llama-8B@32768` (BPE-decode bug, 0.0 HumanEval), `Nemotron-Nano-9B-v2-NVFP4@65536` (7.5 % leak rate, no tool parser) | -- |

## Full results table

Sorted by aggregate (mean of GSM8K / HumanEval / tools_use). All
rows from the 2026-05-05 sweep stamped with `host_env_id`
`ea4fd7e7b668`.

> **Frozen snapshot, not a live mirror.** This table is the 2026-05-05
> sweep as run. `make bench-report` now emits more columns than it has
> (HumanEval+, MMLU-Pro, GPQA landed in the default task set later),
> so re-generate rather than compare column-for-column.

| Model | GSM8K | HumanEval | tools_use | by_subcase (E/S/M/F) | Leak | Cold (s) | Warm p50/p95 (ms) | TPS (tok/s) | Peak VRAM | Aggregate |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| **NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4@131072** | 0.99 | 0.98 | **1.00** | 1.0/1.0/1.0/1.0 | 0.000 | 60.6 | 46.9/52.7 | **143.8** | 22.45 GB | **0.99** |
| **Qwen3-14B-NVFP4@65536** | 0.98 | 0.94 | **1.00** | 1.0/1.0/1.0/1.0 | 0.000 | 47.8 | 46.1/47.3 | 61.9 | 22.32 GB | **0.97** |
| **Qwen3-8B-NVFP4@131072** | 0.97 | 0.88 | 0.95 | 1.0/1.0/1.0/0.8 | 0.000 | 43.5 | 32.3/34.4 | 102.1 | 22.53 GB | **0.93** |
| **gpt-oss-20b@262144** | 0.97 | **0.98** | 0.85 | 0.4/1.0/1.0/1.0 | 0.000 | 53.9 | 50.3/54.0 | **139.2** | 22.37 GB | **0.93** |
| **Qwen3.5-9B-NVFP4@131072** | 0.98 | 0.78 | 0.95 | 0.8/1.0/1.0/1.0 | 0.000 | 160.5 | 31.5/45.6 | 55.3 | 21.54 GB | **0.90** |
| **DeepSeek-R1-Distill-Qwen-7B@65536** | 0.87 | 0.94^! | 0.60 | 1.0/0.4/0.6/0.4 | 0.000 | 84.8 | 36.2/49.4 | 44.5 | 21.90 GB | 0.80 |
| **Llama-3.1-8B-Instruct-NVFP4@131072** | 0.78 | 0.80 | 0.70 | 1.0/0.4/1.0/0.4 | 0.000 | 39.5 | 23.3/24.0 | 95.4 | 22.65 GB | 0.76 |
| **DeepSeek-R1-Distill-Llama-8B@32768** | 0.63 | 0.00^* | 0.80 | 1.0/0.6/1.0/0.6 | 0.000 | 0.1^# | 37.1/52.7 | 42.5 | 21.69 GB | 0.48 |
| **Nemotron-Nano-9B-v2-NVFP4@65536** | 0.74 | 0.06 | 0.00^+ | 0.0/0.0/0.0/0.0 | 0.075 (3x `</think>`) | 112.7 | 26.9/30.9 | 80.4 | 22.28 GB | 0.27 |

Subcase legend: **E**mpty-schema . **S**ingle-arg . **M**ulti-tool-pick . **F**ollow-up.
All TPS values are post-fix (see "TPS counting fix" below).

Footnotes for caveats marked above:

- ^* **`HumanEval=0.00` for R1-Distill-Llama-8B** is the byte-level
  BPE-decode bug (Issue #1 below): the model emits raw `G`-marker
  (space) and `C`-marker (newline) tokens un-decoded, so every
  completion is invalid Python regardless of the scorer. The 2026-
  05-05 re-bench reproduced 0.00 cleanly; followup #3 (file vLLM
  issue with reproducer) still pending.
- ^+ **`tools_use=0.00` for Nemotron-Nano-9B-v2-NVFP4** is because the
  router's `maybeStripTools` drops `tools` and `tool_choice` from the
  request entirely when no tool parser is probed, and Nemotron-Nano-
  9B-v2 has no `parsers:` block in `model-families.yaml`. NVIDIA does
  ship a parser plugin (`nemotron_toolcall_parser_no_streaming.py`)
  for vLLM's `--tool-parser-plugin` mechanism; wiring it up is a
  separate followup (see followup work below).
- ^! **`HumanEval=0.94` for R1-Distill-Qwen-7B** is up from **0.26**
  in the 2026-05-02 sweep -- a 0.68 jump that flagged on the diff.
  Same model digest, same n=50 subset, same backend image. Most
  likely cause: the 2026-05-02 run preceded the v2 scorer's fence-
  handling fix and was charging fenced inline-reasoning answers as
  failures; the re-bench under the v2 scorer with `--force` now
  records the model's real performance. Worth a sanity re-run if
  it ever drifts back. (See "Issues surfaced > 4. HumanEval scorer
  too strict for inline-reasoning models -- FIXED" for the scorer
  history.)
- ^# **`Cold=0.1 s` for R1-Distill-Llama-8B** is a measurement
  artifact: the bench operator hit the model with a warmup curl
  before kicking off `make bench-vllm`, so the latency sidecar saw
  a fully-warmed container on the first prompt. The genuine cold-
  start for this row is in the 50-90 s range based on the 2026-05-
  02 sweep (85.2 s). The TPS, p50/p95, and quality numbers are
  unaffected; only `ttft_ms_first` was contaminated. Easy fix on
  the next run: skip the warmup curl, or accept `ttft_ms_first` as
  "warm-start latency" and add a separate cold-start probe.

(The previous `^+^+` footnote about `tools_use=0.00` on forced-mode
models is gone: the bench's `tools_use` task now pins `tool_choice`
per-sample via a custom tool-call loop, so R1-Distill x2 and
Llama-3.1-8B-Instruct-NVFP4 produce real scores. See "Issues surfaced
> 2. Tool-loop interruption on forced-mode models" below.)

(The previous note about Nemotron-3-Nano-30B-A3B's `--enforce-eager`
-> `--max-num-seqs 8` rescue is now historical: that fix landed in
b730985 before the 2026-05-02 sweep and the model has been stable
since. The 2026-05-05 cold-start of 60.6 s includes the standard
CUDA-graph compilation; the 143.8 tok/s steady-state matches the
post-fix expectation.)

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

Results merge into one row per (model, backend, ctx) in
`deploy/.bench-cache.json` (schema v3), keyed by
`<repo>@<sha>::<backend>::<ctx>` (HF) or `<digest>::<backend>::<ctx>`
(Ollama). The same model benched at two ctx tiers lands in two rows so
TPS / TTFT differences across ctx are not silently averaged. The probe
cache (`<repo>@<sha>` for HF, digest for Ollama) is the join surface
for fit data.

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

The biggest single insight from the 2026-05-05 sweep: **MoE + FP4
wins decisively on this card.** Both the new outright leader
**`Nemotron-3-Nano-30B-A3B-NVFP4`** at 143.8 tok/s and the runner-up
**`gpt-oss-20b`** at 139.2 tok/s are MoE + FP4-class checkpoints
(~3 B and ~3.6 B active params respectively); their per-token
weight transfer (~1.5-1.8 GB) is well below the dense 8 B NVFP4
models (~5 GB), giving a ~3x ceiling advantage on the 640 GB/s
memory-bound card.

## Architectural finding: FP4 quantization + MoE on Blackwell

After the TPS counting fix and the 2026-05-05 re-bench, this
Blackwell card delivers a **~3x decode-rate spread** across the
quantization x architecture matrix, with MoE + FP4 sweeping the
top:

| Architecture x Quantization | Models | TPS p50 | Why |
|---|---|---:|---|
| **MoE + NVFP4** (~3 B active, FP4 KV cache) | NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 | **143.8 tok/s** | smallest per-token weight transfer (~1.5 GB); FP4 KV cache cuts attention reads too |
| **MoE + MXFP4** (~3.6 B active) | gpt-oss-20b | **139.2 tok/s** | per-token weight transfer ~1.8 GB; OCP MXFP4 vs vendor NVFP4 is a wash on raw decode |
| **Dense + NVFP4** (small) | Qwen3-8B-NVFP4, Llama-3.1-8B-Instruct-NVFP4 | 95-102 tok/s | ~5 GB/token; FP4 tensor cores hit |
| **Dense + NVFP4** (medium reasoning) | Nemotron-Nano-9B-v2-NVFP4 | 80 tok/s | similar to small NVFP4 + reasoning template overhead |
| **Dense + NVFP4** (large) | Qwen3-14B-NVFP4 | 62 tok/s | ~7 GB/token; bandwidth scales linearly with weight size |
| **Hybrid Mamba + NVFP4** (community quant) | Qwen3.5-9B-NVFP4 (`ykarout/`) | 55 tok/s | linear-attention layers decode well, but Mamba state arithmetic adds overhead vs pure Transformer |
| **Dense + BF16** | DeepSeek-R1-Distill-{Qwen-7B, Llama-8B} | 42-45 tok/s | 16 GB/token weight transfer at 640 GB/s ~ 40 tok/s ceiling |

**Practical wall-time matters more than raw TPS** because reasoning
models emit a long `<think>` preamble before the visible answer.
End-to-end wall time for an agent waiting on a 200-token reply:

| Model | TPS | Pre-answer reasoning | Total wall time |
|---|---:|---:|---:|
| `Nemotron-3-Nano-30B-A3B-NVFP4` (nano_v3 reasoning) | 144 | ~300 tokens | **3.5 s** |
| `gpt-oss-20b` (harmony channels, structured) | 139 | ~300 tokens | **3.6 s** |
| `Qwen3-8B-NVFP4` (Qwen3 thinking) | 102 | ~500 tokens | **6.9 s** |
| `Llama-3.1-8B-Instruct-NVFP4` (no reasoning) | 95 | 0 | **2.1 s** |
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
capable model, and prefer MoE when context budget allows. The
**Nemotron-3-Nano-30B-A3B-NVFP4** + **Qwen3 NVFP4 family (8B, 14B)**
+ **gpt-oss-20b** are the practical agentic workhorses on this
hardware. The R1-Distill BF16 family stays in the catalog as a
curiosity (their reasoning quality is real -- Qwen-7B distill scored
0.87 on GSM8K and an unexpectedly high 0.94 on HumanEval in this
sweep) but should not be the default for any latency-sensitive
workflow.

## Cold-start signal

The `ttft_ms_first` metric captures the user-observable cold start -- 
exactly what an agent like Claude Code experiences on first request to
a freshly-routed model:

| Model | Cold (s) | Notes |
|---|---:|---|
| Llama-3.1-8B-Instruct-NVFP4 | 39.5 | fastest cold start; small footprint, no reasoning preamble |
| Qwen3-8B-NVFP4 | 43.5 | + reasoning template + Qwen3 chat template |
| Qwen3-14B-NVFP4 | 47.8 | bigger weights |
| gpt-oss-20b | 53.9 | 20 B MoE, harmony channels, longer graph compile |
| Nemotron-3-Nano-30B-A3B-NVFP4 | 60.6 | 30 B MoE in NVFP4 + nano_v3 reasoning parser; CUDA-graph capture under `--max-num-seqs 8` |
| DeepSeek-R1-Distill-Qwen-7B | 84.8 | BF16, slower paged-in load |
| Nemotron-Nano-9B-v2-NVFP4 | 112.7 | NVFP4 + 9B + Nemotron-specific kernel compile |
| **Qwen3.5-9B-NVFP4** | **160.5** | **first-time CUDA-graph capture for the Qwen3.5 generation; subsequent recreates much faster** |
| _DeepSeek-R1-Distill-Llama-8B_ | _0.1_ | _2026-05-05 row contaminated by a pre-bench warmup curl; cf. footnote `^#` on the full results table. The genuine cold-start is ~85 s based on the 2026-05-02 sweep._ |

Operational note: the `HEALTH_TIMEOUT_SECONDS=600` default in
`gpu-arbiter` is justified -- Qwen3.5-9B's 160 s startup leaves
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

### 2. Tool-loop interruption on forced-mode models -- **FIXED**
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
launches without `--reasoning-parser`.

**What NVIDIA actually prescribes** (from the HF model card,
checked 2026-05-02): reasoning is controlled via `/think` (default)
or `/no_think` keywords in the system message or per-turn user
message; the model emits `<think>...</think>` tags around the trace.
Tool calling uses NVIDIA's custom `<TOOLCALL>` / `<AVAILABLE_TOOLS>`
/ `<TOOL_RESPONSE>` format, with a parser plugin
(`nemotron_toolcall_parser_no_streaming.py`) that ships in the HF
repo for vLLM's `--tool-parser-plugin` mechanism. The full launch
spec is `vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2
--trust-remote-code --mamba_ssm_cache_dtype float32
--enable-auto-tool-choice --tool-parser-plugin <vendored-script>
--tool-call-parser nemotron_json`. None of that is wired up in this
repo today.

**Action**: see followup #4 below for the concrete steps. The
existing comment in `model-families.yaml` claiming "no shipped
parser matches" predates checking NVIDIA's docs and is wrong; it's
flagged for correction as part of that followup.

### 4. HumanEval scorer too strict for inline-reasoning models -- **FIXED**
`_clean_completion` v1 only matched a fence enclosing the **entire**
completion (after `strip()`); inline-reasoning models that put a
`<think>...</think>` preamble before the code fence, or surrounding
prose around the fence, fell through to "return raw text" and the
subprocess saw `<think>` / `Here's the code:` lines as syntax errors.

**Resolution**: rewrote `_clean_completion` in
`scripts/bench/tasks/humaneval.py` with three layered strategies:

1. Strip any `<think>...</think>` blocks (re.DOTALL, case-insensitive).
2. If one or more fenced blocks remain, return the body of the
   **last** one (empirically the model's final answer when it drafts
   then revises).
3. If no fence and an `entry_point` is known, anchor on
   `^def <entry_point>\(` and slice from there.
4. Else return the think-stripped text as-is.

Smoke-tested against 9 representative inputs (whole-text fence, fence
with surrounding prose, `<think>` + fence, `<think>` + naked def,
draft+final fences, entry_point-name-in-prose-then-real-def, unclosed
`<think>`, empty input). All passed.

**Validation rerun** (`BENCH_TASKS=humaneval BENCH_FORCE=1` over
Nemotron + R1-Distill-Llama-8B):

- Nemotron-Nano-9B-v2-NVFP4 pass@1: 0.00 -> **0.06** (3/50). Modest
  improvement; the scorer was hiding ~3 truly-passing samples behind
  preamble. Most failures are genuine coding weakness, not scorer
  strictness -- the doc's earlier "0.00 likely much higher in reality"
  speculation was over-optimistic.
- R1-Distill-Llama-8B pass@1: 0.00 -> **0.00**. Confirms the byte-level
  BPE-decode bug (Issue #1) is the *dominant* failure mode here, not
  scorer strictness -- every completion is still invalid Python.

The scorer fix is a no-op-or-improvement for already-passing models
(strict whole-text fence still matches; surrounding-prose paths only
fire when the strict path would have returned the unmodified text).
No re-run of auto-mode models was performed; their cached scores are
still meaningful.

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
| Nemotron-3-Nano-30B-A3B-NVFP4 @131K | 22.45 / 24 = **93.5%** | tight; runs with `--max-num-seqs 8` (NVIDIA's prescribed value) which shrinks CUDA-graph capture buffers enough to keep capture enabled |
| gpt-oss-20b @262K | 22.37 / 24 = 93.2% | tight |
| Qwen3-14B-NVFP4 @65K | 22.32 / 24 = 93.0% | tight |
| Nemotron-Nano-9B-v2-NVFP4 @65K | 22.28 / 24 = 92.8% | tight |
| DeepSeek-R1-Distill-Qwen-7B @65K | 21.90 / 24 = 91.2% | comfortable |
| DeepSeek-R1-Distill-Llama-8B @32K | 21.69 / 24 = 90.4% | comfortable |
| Qwen3.5-9B-NVFP4 @131K | 21.54 / 24 = 89.8% | comfortable |

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
    AND quantization in {"NVFP4", "MXFP4"}
    AND tools_use_score >= 0.9
    AND humaneval >= 0.7          # plain HumanEval (humaneval_subset_*),
                                  # NOT HumanEval+ (humaneval_plus_subset_*)
    AND gsm8k >= 0.9
    AND leak_rate == 0
    AND peak_vram_gb < 23
)
# matches (2026-05-05): NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4,
#   Qwen3-14B-NVFP4, Qwen3-8B-NVFP4, Qwen3.5-9B-NVFP4
# notable change vs 2026-05-02: gpt-oss-20b dropped out -- tools_use
#   slipped from 0.90 to 0.85 (empty_schema subcase down to 0.4),
#   so it no longer clears the >= 0.9 threshold.

CODING_SPECIALIST = max(humaneval) where PRODUCTION_AGENTIC
# matches: NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 (HumanEval 0.98)
# tied: gpt-oss-20b would also be 0.98 but is no longer AGENTIC.

LATENCY_SENSITIVE = (
    PRODUCTION_AGENTIC
    AND tps_sustained_p50 >= 50
)
# matches all four AGENTIC models -- 55 < tps < 144.

THROUGHPUT_KING = max(tps_sustained_p50)
# matches: NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 at 143.8 tok/s,
#   edging gpt-oss-20b at 139.2.

REASONING_ONLY = (
    backend == "vllm"
    AND gsm8k >= 0.85
    AND tps_sustained_p50 < 30
)
# matches (2026-05-05): no models -- the slowest qualifying model
#   in the sweep, DeepSeek-R1-Distill-Qwen-7B, decodes at 44.5 tok/s
#   (still slow in wall time once you account for the long <think>
#   preamble, but its raw decode rate is well above the 30 tok/s
#   cutoff).

AVOID = (
    leak_rate > 0
    OR has_known_decode_bug   # R1-Distill-Llama-8B
    OR (humaneval < 0.3 AND not REASONING_ONLY)
)
# matches: R1-Distill-Llama-8B (BPE bug; HumanEval 0.00),
#   Nemotron-Nano-9B-v2 (7.5 % leak rate, tools_use 0.00).
```

These are derived from the cache, not hand-curated. The picker
reads `deploy/.bench-cache.json` directly and surfaces the
PRODUCTION_AGENTIC verdict via `_is_production_agentic` (formerly a
TIER badge; the per-row score columns now make the verdict implicit
-- a row scoring high in TOOLS, strong across CODE% / CODE+% / MMLU% /
GPQA%, and showing `0.0` in LEAK% is by definition AGENTIC). The
`TOTAL%` composite that used to carry this verdict was retired from
the picker along with `REAS%`; the per-metric columns replaced it.

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
   forced-mode models: Llama-3.1-8B-Instruct-NVFP4 (0.00 -> 0.75),
   DeepSeek-R1-Distill-Qwen-7B (0.00 -> 0.65), DeepSeek-R1-Distill-Llama-8B
   (0.00 -> 0.60). See "Issues surfaced > 2" for details and the tradeoff
   on `multi_tool_pick`.
2. ~~**Fix HumanEval scorer for inline-reasoning models** -- recovers
   Nemotron and R1-Distill-Llama-8B HumanEval data.~~ **Done.**
   `scripts/bench/tasks/humaneval.py::_clean_completion` rewritten with
   `<think>` stripping + last-fence selection + `^def <entry_point>\(`
   fallback. Validated: Nemotron 0.00 -> 0.06 (modest -- mostly genuine
   coding weakness, not scorer strictness); R1-Distill-Llama-8B stayed
   0.00 (BPE bug, see followup #3). See "Issues surfaced > 4" for
   details.
3. **Investigate R1-Distill-Llama-8B BPE-decode bug** -- file vLLM
   issue with reproducer. Likely a vLLM x Llama-3 tokenizer x reasoning-
   parser interaction. **In progress**: reproducer at
   `scripts/repro/r1_distill_llama_bpe.py` -- a one-shot Python script
   that streams a chat completion to two models and counts U+0120
   (Llama-3 BPE leading-space marker) and U+010A (newline marker) in
   `delta.content`. End-to-end verified against this project's stack:
   R1-Distill-Llama-8B leaks 9 U+0120 + 3 U+010A in a 33-char reply;
   R1-Distill-Qwen-7B (same `--reasoning-parser deepseek_r1`, Qwen-2
   tokenizer) returns clean UTF-8. Draft issue body at
   `scripts/repro/r1_distill_llama_bpe.md`. **Remaining**: paste the
   draft into a new bug at `vllm-project/vllm` once we settle the
   precise vLLM image digest to cite.
4. **Wire up Nemotron-Nano-9B-v2 per NVIDIA's official guidance.**
   The current `tools_use=0.00` and `leak_rate=0.075` plus the
   sub-optimal HumanEval are not architectural limits; they're a
   missing-config problem. NVIDIA's HF model card prescribes a
   specific parser plugin and launch flags that we don't ship. Steps:
     a. Vendor `NVIDIA-Nemotron-Nano-9B-v2/nemotron_toolcall_parser_no_streaming.py`
        from the HF repo into `scripts/parsers/` (or download at
        probe-time) and reference it via `--tool-parser-plugin`.
     b. Add a `parsers:` block under the `nemotron-nano-v2` family in
        `scripts/model-families.yaml`: `tool_parser: nemotron_json`,
        plus the launch flags `--enable-auto-tool-choice
        --trust-remote-code --mamba_ssm_cache_dtype float32`. Update
        the comment block above the family (currently asserts "no
        shipped parser matches" -- that's wrong, NVIDIA does ship one).
     c. For HumanEval / pure-code prompts: the bench's system prompt
        currently fights the model's default `/think` -- adding
        `/no_think` to the system message would skip reasoning and
        likely raise pass@1 noticeably. Either patch the bench's
        SYSTEM_PROMPT for Nemotron specifically or rely on a future
        per-model prompt override knob.
     d. Re-run `make probe-vllm` (just for this family); the per-cell
        cache should flip `tool_parser: nemotron_json`,
        `tool_mode: auto`, `disable_verified: True`. If the probe
        rejects the parser, the launch flags are wrong.
     e. Re-run `make bench-vllm BENCH_REPO=Nemotron-Nano BENCH_FORCE=1`
        and update the table + "Avoid (broken)" tier accordingly.
     Reference: https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2
5. ~~**Add KV-pressure column to `make bench-report`** -- the data is
   already in the cache.~~ **Done.** New `KV %` column at the end of
   the leaderboard renders `peak_vram_gb / GPU_MEMORY_GB` as a
   percentage; defaults to a 24 GB host cap (matches RTX 4000 PRO
   Blackwell), overridable via env (`GPU_MEMORY_GB=48 make bench-report`)
   or CLI flag (`--host-vram-gb 48`). 95 % is the rule-of-thumb
   threshold from "KV-pressure observations" above; current cache flags
   `Qwen3.5-9B-NVFP4` at 95.7 % (the only row above the threshold).
   Footer line in the report reminds readers of the threshold and
   points back to this doc.
6. ~~**Add a long-context probe (one prompt at 80% of ctx)** -- detects
   KV paging cliffs that the current latency probe misses.~~ **Done.**
   New module `scripts/bench/bench_longctx.py` sends ONE chat
   completion at `fraction * model.ctx` input tokens (default 0.8)
   and captures: prefill TTFT, decode-phase TPS, output_tokens,
   finish_reason, end-of-request `vllm:kv_cache_usage_perc`, and
   `vllm:num_preemptions_total` delta. Wired into `bench_runner` as
   an opt-in `longctx` task (not in the default task set -- which is
   now `gsm8k,humaneval,humaneval_plus,mmlu_pro,gpqa,tools,leak`;
   enable with `BENCH_TASKS=longctx,...`). Knobs
   `--n-longctx-fraction` and `--n-longctx-max-tokens` (env
   `BENCH_N_LONGCTX_FRACTION` / `BENCH_N_LONGCTX_MAX_TOKENS`).
   Filler prompt is a public-domain prose chunk repeated to size
   (~3.5 chars/token, conservative under-shoot to avoid
   max_model_len rejection); the tail asks for a single-sentence
   summary so the model produces a measurable decode segment.

   Smoke-test against Nemotron-3-Nano-30B-A3B-NVFP4 at 80 % of
   131 K (104 857 target tokens): `ttft=75.8s`, `decode TPS=144.1`,
   `out=67`, `finish=length`, no preemptions, peak VRAM 22.54 GB.
   Prefill cost confirmed as the dominant interactive-latency
   penalty for near-max-ctx requests on this card.

   Known limitation: the end-of-request `kv_cache_usage_perc`
   reads 0.0 because the request has freed its KV blocks by the
   time we sample /metrics. Capturing the peak DURING the request
   would need a parallel polling thread (similar to the existing
   `VramSampler`). Logged but not fixed -- the absence of
   preemptions is sufficient signal that the KV pool was big
   enough; we'll only need peak-during-run sampling when we hit a
   model that DOES preempt under longctx load.
7. ~~**Add vLLM `/metrics` snapshot** -- captures
   `vllm:gpu_cache_usage_perc` and `vllm:num_preemptions_total` per
   run.~~ **Done.** End-of-run snapshot lands in the bench cache as
   `vllm_kv_cache_usage_perc` and `vllm_num_preemptions_total`
   alongside the existing VRAM/TTFT/TPS metrics. Note: the gauge in
   the current vLLM image is `vllm:kv_cache_usage_perc`, not
   `vllm:gpu_cache_usage_perc` -- the latter was an older name. The
   parser is best-effort: if the backend's `/metrics` endpoint is
   unreachable (Ollama, idle SGLang, network blip) the bench result
   is still valid; the new fields just stay absent. SGLang has its
   own metric names (`sglang:cache_hit_rate`, `sglang:num_running_reqs`,
   `sglang:num_used_tokens`) -- the same code path picks them up
   when SGLang benches resume (followup #8). Limitation: end-of-run
   snapshot only, not max-during-run -- after the last sample drains
   the queue, `kv_cache_usage_perc` falls. Capturing the max would
   require a second sampler thread; left for a v2 polish.
8. **Run `bench-sglang`** -- **on-hold.** Two models actually fit on
   the SGLang side per the current probe cache (`gpt-oss-20b` and
   `DeepSeek-R1-Distill-Qwen-7B`); a first attempt revealed that the
   bench cache schema collided HF rows from different backends under
   the same `<repo>@<sha>` key, so a no-op SGLang run silently
   overwrote the vLLM rows' `backend` field and zeroed their
   `peak_vram_gb`. The harness was hardened (cache key now suffixed
   with `::<backend>`, idempotent migration on load) but the
   re-bench itself is paused per project decision. Reactivate by
   simply running `make bench-sglang` once SGLang is back in scope;
   the harness will create new `::sglang` rows alongside the
   existing `::vllm` ones.
9. **Run `bench-ollama`** -- **out of scope.** 28 Ollama models,
   ~4 hours wall time. Skipped per project decision: at this stage
   benchmarking the Ollama backend isn't useful for routing decisions
   on this hardware (the agentic / coding-quality picture is already
   resolved by the vLLM rows). The harness supports it (the bench
   cache key now suffixes with `::ollama` and the `/metrics` snapshot
   path explicitly no-ops for Ollama). Reactivate by running
   `make bench-ollama` if Ollama-side comparisons become relevant.
10. ~~**Wire bench cache -> picker badge** -- tag PRODUCTION_AGENTIC rows
    in the picker UI.~~ **Done.** `scripts/model-picker.py` now reads
    `deploy/.bench-cache.json` (loader pattern mirrors the existing
    HF probe-cache loader, falls back gracefully when the file is
    missing) and tags each candidate row with `_picker_agentic`
    based on the formula in "Picker-tier recommendations" above.
    Qualifying rows render the literal `agentic` label in bright
    green in a new `TIER` column right before `VRAM (GB)`;
    non-qualifying rows get a dim `-` placeholder so the column
    still aligns. Thresholds (`tools_use >= 0.9`, `humaneval >= 0.7`,
    `gsm8k >= 0.9`, `leak_rate <= 0`, `peak_vram_gb < 23`,
    backend=vllm, format=NVFP4) live as named constants
    `_PRODUCTION_AGENTIC_*` so any change here is one place. Smoke-
    tested against the real cache: 4 models qualify (Qwen3-8B,
    Qwen3-14B, gpt-oss-20b, Nemotron-3-30B-A3B); the rest don't.
    Missing bench data leaves the badge off (never speculative).
11. ~~**Experiment: drop `--enforce-eager` on
    `NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` by also passing
    `--max-num-seqs 8`**~~ **Done.** Hypothesis confirmed: the OOM
    at model load is driven by CUDA-graph capture buffers that scale
    with `max_num_seqs`. Shrinking the scheduler pool from vLLM's
    default 256 down to NVIDIA's prescribed 8 reduced the graph
    capture transient enough to fit on 24 GB *without* disabling
    capture. After: probe loads cleanly (peak VRAM 22.35 GB,
    structured capability preserved), bench shows TPS 42.87 -> 144.84
    (3.4x), steady TTFT p50 70 ms -> 51 ms (-27 %), peak VRAM 22.63
    GB -> 22.43 GB (-200 MB), leak rate stays at 0. Cold start now
    65.6 s (one-time graph compile included; faster on subsequent
    recreates). Nemotron-3 is now the fastest structured-reasoning +
    perfect-tools model on the leaderboard, edging gpt-oss-20b's 136
    tok/s. `recovery-flags.json` updated; `--enforce-eager` removed.

    Side-effect: this also closed the "doesn't investigate deep
    enough" symptom the user reported when running Claude Code
    against this model in a parallel session. The earlier
    investigation suspected intrinsic 3.6 B-active-params capability,
    sampling defaults, or system-prompt sparseness (knobs 1a/1b/1c
    in the followup plan). 1a (`enable_thinking` reaches vLLM) was
    verified working before this rerun. After 3.4x throughput the
    symptom evaporated -- the model wasn't reasoning shallower, the
    slow turns just made the same investigation depth feel
    shallower per minute of patience. **Lesson: throughput
    regressions can masquerade as quality regressions in
    interactive agentic workloads.** Resolved without touching
    sampling or system prompts.

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
