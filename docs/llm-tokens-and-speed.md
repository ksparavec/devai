# LLM tokens and inference speeds -- a beginner's guide

This page is a companion to
[`nvfp4-number-formats.md`](nvfp4-number-formats.md) and
[`nvfp4-coldstart.md`](nvfp4-coldstart.md). It covers two things that
trip up almost everyone reading benchmark numbers for the first time:

1. **What an LLM token actually is** -- and why it is *nothing like* the
   tokens a programming-language compiler produces.
2. **Why "ingest a prompt" and "generate a reply" run at completely
   different speeds**, and how the GPU's memory bandwidth (not its
   compute throughput) is usually the binding constraint.

The worked examples use **`nvidia/Qwen3-8B-NVFP4`** and the
Qwen3.8-27B builds on this project's **NVIDIA RTX PRO 4000 Blackwell
(24 GB GDDR7)** workstation card. Numbers are of three kinds and each
is labelled: **measured** here (with its definition, number of
requests and runs, and date), **derived** from a model whose
assumptions are stated, or **external** rules of thumb and
literature. [statistics-primer.md](statistics-primer.md) Sec. 8, 9 and
14 explain how the measured ones are reported; most are single runs
(Sec. 1 of the primer).

If you already know what BPE is and you can derive the
memory-bandwidth ceiling on a model in your head, you can skip to Sec. 6.

---

## 1. What is a "token", actually?

If you have a compiler background, the word *token* probably means
something specific: a syntactic unit produced by a lexer, like
`if`, `(`, `x`, `==`, `42`, `)`, `{`. Each token corresponds to a
language construct chosen by the language designer. Two source files
that mean the same thing (modulo whitespace) tokenise the same way.
This is **lexer tokenisation** -- a deterministic, designed split.

LLM tokens are a completely different beast. They are the output of a
**learned, statistical compression** of UTF-8 text. There is no
designer; there is a training corpus, a vocabulary size, and an
algorithm that picks subword units which appear *frequently* in the
training data. The tokeniser does not understand syntax, words, or
languages. It just knows that some sequences of bytes occur often
enough to deserve their own slot in the vocabulary.

A few consequences worth internalising up-front:

- **A token is not a word.** Sometimes it is one whole word
  (`"hello"`). Often it is a fragment (`"un"`, `"believ"`, `"able"`),
  a single character, or even a single byte.
- **A token includes its surrounding whitespace.** `"hello"` and
  `" hello"` are *different* tokens with *different* IDs.
- **Two LLMs do not share tokens.** A Llama tokeniser and a Qwen
  tokeniser produce different token IDs for the same string and
  often a different *number* of tokens.
- **Token count is what you pay for.** API pricing, context windows,
  KV-cache memory, decode latency -- all of these are denominated in
  tokens, not characters.

The Qwen3 tokeniser used by `Qwen3-8B-NVFP4` has a vocabulary of
**151 936** tokens (visible in `config.json` as `vocab_size`). That's
the entire universe of "things this model can emit one of" at any
given step.

---

## 2. How a BPE tokeniser actually works

The dominant algorithm in modern LLMs is **Byte-Pair Encoding (BPE)**,
originally a 1994 data-compression trick adapted to NLP by Sennrich,
Haddow & Birch (2016 -- *Neural Machine Translation of Rare Words with
Subword Units*, ACL).

The training-time algorithm is short:

1. Start with a vocabulary of all individual bytes (256 entries) or
   all individual Unicode characters (a few thousand).
2. Tokenise a large training corpus using just those base units.
3. Find the **most frequent adjacent pair** of tokens across the
   corpus. Add a new token that represents that pair, and rewrite
   every occurrence in the corpus to use the new merged token.
4. Repeat step 3 until the vocabulary reaches the target size
   (typically 30 K - 150 K).

The result is an ordered list of **merge rules** plus a vocabulary
mapping. Tokenising any new string at inference time means greedily
applying the merge rules in the order they were learned.

### Worked example -- building tiny BPE in 5 steps

Suppose your entire training corpus is the string `low low low lowest newest`. Start with character-level tokens:

```
   l  o  w  _  l  o  w  _  l  o  w  _  l  o  w  e  s  t  _  n  e  w  e  s  t
```

(`_` represents a space.) Now look for the most frequent adjacent
pair. Every step:

- **Step 1**: pair `l o` appears 4 times -> merge into new token `lo`.
  Corpus becomes:

  ```
     lo w _ lo w _ lo w _ lo w e s t _ n e w e s t
  ```

- **Step 2**: pair `lo w` appears 4 times -> merge into `low`.

  ```
     low _ low _ low _ low e s t _ n e w e s t
  ```

- **Step 3**: pair `e s` appears 2 times -> merge into `es`.

  ```
     low _ low _ low _ low es t _ n e w es t
  ```

- **Step 4**: pair `es t` appears 2 times -> merge into `est`.

  ```
     low _ low _ low _ low est _ n e w est
  ```

- **Step 5**: pair `n e` appears 1 time, pair `w est` appears 1 time,
  etc. Keep going until you hit your vocabulary budget.

The vocabulary is now `{l, o, w, _, e, s, t, n, lo, low, es, est}`.
Tokenising a *new* string like `lowest newest` runs the same merge
rules in order:

```
   l o w e s t _ n e w e s t
   -> lo w e s t _ n e w e s t      (merge 1)
   -> low e s t _ n e w e s t       (merge 2)
   -> low es t _ n e w es t         (merge 3)
   -> low est _ n e w est           (merge 4)
```

Result: `["low", "est", "_", "n", "e", "w", "est"]` -- 7 tokens for
13 characters. The merges learned during training make the common
sequence `low est` cheap (2 tokens) while the rare sequence `n e w`
stays expensive (3 tokens).

Real LLM tokenisers do this same thing on a multi-billion-character
corpus with vocabularies in the 30 K - 150 K range. The result is
that frequent words and word fragments collapse to single tokens,
while rare strings split into many short tokens.

> **Want to see this live?** Andrej Karpathy's *Let's build the GPT
> Tokenizer* (YouTube, 2024) walks through a from-scratch BPE
> implementation and visualises the merges interactively -- easily
> the best educational resource on the topic.

---

## 3. Token surprises -- things that catch newcomers off-guard

### 3.1 Whitespace is part of the token

`"hello"` and `" hello"` (with a leading space) are **different
tokens** in every modern tokeniser. The leading space is glued to the
*following* word. This is why the GPT-style tokenisers refer to
"tokens with a leading space" using a special prefix character (Unicode
U+0120, often shown as a capital G with an overdot in vocab files) --
that prefix is the visualisation of a leading space.

### 3.2 Numbers usually split character-by-character

The string `"12345"` is rarely one token. Most tokenisers split it
into `["1", "23", "45"]` or even `["1", "2", "3", "4", "5"]`. Why?
Because the training corpus contains every possible number, and no
single multi-digit string is frequent enough to earn its own merge.
This has practical consequences: arithmetic-heavy prompts use far
more tokens than their character count would suggest, and LLMs that
"can't do arithmetic" are partly bottlenecked by their tokeniser.

### 3.3 Code can be very dense or very sparse

Common identifiers and keywords (`function`, `return`, `console.log`,
`import numpy as np`) often collapse to 1-2 tokens because they
appeared millions of times in the training set. Rare or
domain-specific names (`def my_obscure_helper_function`) can take
many more tokens than their character count suggests.

### 3.4 Non-English text uses far more tokens per word

A tokeniser trained mostly on English will assign single-token slots
to common English words but encode Chinese, Arabic, Korean, or
Cyrillic strings as long sequences of byte-level tokens. The same
sentence can take 1.3 tokens per word in English, 3-5 tokens per word
in German, and 6+ tokens per word in many non-Latin scripts. Modern
multilingual tokenisers (Qwen3 included -- its 151 K vocab is much
larger than English-only models because most of the additional slots
are CJK characters) reduce but do not eliminate this gap.

### 3.5 Special tokens carry chat-template structure

Modern chat-tuned models reserve token IDs for *control tokens* that
have no character representation in normal text. Qwen3 reserves
slots like `<|im_start|>`, `<|im_end|>`, `<|tool_call|>`,
`<|reasoning|>`. The chat template (e.g. `tokenizer_config.json`)
inserts these around your messages so the model can tell where each
turn starts and stops.

This is why the bench harness explicitly looks for these markers in
the *output* -- if the model emits `<|im_end|>` as visible text instead
of using it as a control token, the parser failed (see
[`bench-results.md`](bench-results.md) "Issues surfaced", #3, for a
real example of `</think>` leaking from `Nemotron-Nano-9B-v2-NVFP4`).

---

## 4. Rough conversion rates -- tokens per word, characters, etc.

Useful rules of thumb for English text on Latin-vocab tokenisers
(external, not measured here):

| Quantity | Approximate ratio | Source |
|---|---|---|
| Characters per token | ~4 | OpenAI's tokeniser FAQ; consistent across BPE models |
| Tokens per English word | ~1.3 | empirical on Common Crawl |
| Words per token | ~0.75 | inverse of above |
| Tokens per page of single-spaced English | ~500 | rough |
| Tokens per typical chat turn (one user message) | 10 - 100 | user chat data |

**Caveat -- these break for code, JSON, numbers, and non-English text**
by factors of 2-5. The bench harness's prompts (
`scripts/bench/data/latency_prompts.jsonl` ) are short English
factual questions like *"What is 2 + 2?"* (14 chars ~ 5 tokens) and
*"Name the largest planet in the Solar System. One word."* (54 chars
~ 14 tokens).

**Why this matters for the numbers in this repo.** Every
tokens-per-second value in the bench cache counts tokens as
**characters / 4** of the streamed text (reasoning included), not as the
engine's own count. Prompt sizes built with a characters-per-token rule
go wrong too: the long-context prompts in Sec. 8 came out at 0.75 of
their nominal token count, and the concurrency tool's repetitive filler
text reached about 6.2 characters per token. Always ask which counter a
"tokens" number used ([statistics-primer.md](statistics-primer.md)
Sec. 9).

> **Where the characters/4 count comes from, and how close it is.** The
> bench's streaming request does not set
> `stream_options.include_usage`, and vLLM, SGLang and Ollama (checked in
> the v0.34.2 source) send no usage block without it, so the harness
> falls back to characters/4 on every backend. The only check possible on
> the retained data compares summaries, not per-request counts: the
> harness p50 over all 40 prompts (client timing, characters/4) against
> the engine's median per-request rate (engine token counts, engine
> timing) in the same four runs (Qwen3.8 tokenizer, 2026-09-21/22). The
> ratio was 0.93-1.01 against the engine median over all prompts and
> 0.99-1.02 against the engine median over outputs of at least 128
> tokens (`scripts/stats/perf_engine_runs.py`). Per-request token counts
> were never compared. On code the estimate read 10-25 % low in one
> ad-hoc comparison (2026-09-22, quoted from the harness fix commit; not
> reproducible from retained data).

---

## 5. The two phases of LLM inference

Generating a reply happens in two sharply different phases. Failing
to distinguish them is the single most common source of confusion
when reading inference benchmarks.

### 5.1 Prefill ("ingest") -- process all input tokens at once

When you submit a prompt of, say, 200 tokens, the runtime tokenises
the prompt and then runs **one forward pass** through the model that
processes *all 200 tokens in parallel*. Inside that single pass:

- Every Linear layer sees a (batch x 200 x hidden) input matrix
  and produces a (batch x 200 x hidden) output.
- Attention runs once over the full 200x200 attention matrix.
- The K and V values for every prompt token get computed and
  written to the KV cache.
- The model emits one logit vector per token, but only the very last
  token's logits matter -- that is the prediction for the next token.

Prefill is **embarrassingly parallel across the sequence dimension**.
Tensor cores get to chew on big matrix multiplies. Memory bandwidth is
amortised over many parallel ops on the same weights. For prompts of
more than a few hundred tokens this phase is **compute-bound** on modern
GPUs; a very short prompt (tens of tokens) is still below that point and
costs about one full read of the weights, like a decode step (Sec. 8).

Prefill speed typically lands in the *thousands of tokens per second*
on a Blackwell card. The one clean measurement here: a 2 958-token
prompt with nothing in the prefix cache reached its first token in
1.84 s on the Qwen3.8-27B W4A16 build and 0.71 s on the NVFP4 build,
i.e. about 1 600 and 4 150 prompt tokens/s (one request each,
2026-09-21; Sec. 8).

### 5.2 Decode ("generate") -- produce one token at a time

Once prefill is done, the runtime enters a loop:

```
   for step in 1..max_new_tokens:
       1. Take last predicted token, embed it, run forward pass
          for sequence length 1.
       2. Read all model weights from VRAM.
       3. Read every K and V from the KV cache (length = prompt + step).
       4. Sample next token from the resulting logits.
       5. Append the new K and V to the cache.
       6. Stream the token back to the client.
```

Each loop iteration produces **one** token. The model's full weights
must travel from VRAM through the on-chip caches to the tensor cores
*for every single token generated*. Worse, the per-token KV-cache
read grows linearly as the sequence lengthens.

Decode is **memory-bandwidth-bound** at batch size 1. The tensor
cores are mostly idle -- by the usual arithmetic-intensity argument
they could process on the order of 100x more matrix work -- but they
have to wait for the next chunk of weights to arrive from VRAM. This
is the central performance reality of single-stream LLM serving.

Decode speed for the reference model was **98-111 tok/s** in this
project's bench (Qwen3-8B-NVFP4, three single runs; Sec. 7 has the
numbers and the math).

### 5.3 Why this matters operationally

- **TTFT** (time-to-first-token) is dominated by prefill: long
  prompt -> long prefill -> user waits longer for the first token.
- **Sustained tok/s** is the decode rate: how fast the reply streams
  *after* the first token. Long generations spend most of their wall
  time here. It is a **per-request** rate; with several requests in
  flight the machine's aggregate throughput rises while each request
  slows down, and the two must not be mixed up
  ([statistics-primer.md](statistics-primer.md) Sec. 9).
- For an interactive agent, low TTFT often matters more than high
  sustained rate; for a batch summariser, the opposite.

---

## 6. The hardware bandwidth hierarchy

LLM inference is, at its heart, a series of memory transfers. Here is
the rough hierarchy of bandwidth available to the GPU, fastest first
(numbers are typical orders of magnitude -- exact figures vary by part):

```
   +-------------------------------------------------------+
   |  GPU register file & L1 cache    ~10 000 GB/s        | on-chip
   |  GPU L2 cache                    ~5 000 GB/s         |
   |  ================================================    |
   |  HBM3e (B100/B200)               ~8 000 GB/s         | off-chip but on-package
   |  GDDR7 (RTX 4000/5090/PRO 6000)  ~650 - 1 800 GB/s   |
   |  ================================================    |
   |  PCIe Gen5 x16 (host <-> GPU)      ~64 GB/s peak,      | system-bus
   |                                  ~55 GB/s practical  |
   |  DDR5 host RAM                   ~50 - 80 GB/s       |
   |  NVMe Gen5 SSD                    ~12 GB/s           | storage
   |  NVMe Gen4 SSD                    ~7 GB/s            |
   |  Gigabit Ethernet                 ~0.1 GB/s          | network
   +-------------------------------------------------------+
```

(Sources: NVIDIA datasheets for Blackwell parts, JEDEC GDDR7 spec,
PCI-SIG PCIe 5.0 spec, NVMe consortium published rates.)

The reference card -- **NVIDIA RTX PRO 4000 Blackwell** -- has 24 GB
of GDDR7. This repo has used **~640 GB/s** as its peak memory
bandwidth, but no primary source for that figure is recorded
([`bench-results.md`](bench-results.md) states it without a citation).
A commonly published figure for this card is 672 GB/s (a 192-bit bus
at 28 Gbps); that has not been verified here. The derivations below
use 640 GB/s and show what changes at 672.

### What each tier matters for

| Tier | When it matters | Example for Qwen3-8B-NVFP4 |
|---|---|---|
| L1/L2 cache | Tiny tensors that fit on-chip -- irrelevant to weights | rare for inference |
| HBM3e / GDDR7 (VRAM) | **Decode bandwidth**: every generated token requires reading all weights | binds decode at about 100 tok/s (Sec. 7) |
| PCIe Gen5 | Moving the model from system RAM to VRAM during cold start | ~0.1 s for 6 GB at PCIe speed alone (derived); the measured weight-load phase took 1.5-1.6 s (Sec. 8) |
| DDR5 RAM | Holding the safetensors mmap before / during page-cache warm-up | ~50 GB/s, rarely the bottleneck |
| SSD / NVMe | First-time load from disk; cold page cache | ~12 GB/s on Gen5 SSD |
| Network | Streaming reply tokens to the client | trivial for text |

The gap between VRAM bandwidth and PCIe bandwidth is roughly
**10 x**. The gap between VRAM and L2 is another **10 x**. This
hierarchy is why a model that lives in VRAM serves at tens of tok/s,
while the same model offloaded to system RAM would serve at a few tok/s
or less and one paged in from disk would be unusable (order-of-magnitude
reasoning, not measured here).

---

## 7. Decode speed math -- the bandwidth ceiling

Now we can connect the dots. The model below is **derived**, and its
assumptions matter ([statistics-primer.md](statistics-primer.md)
Sec. 9, "model-based ceilings"):

- batch 1 (one sequence), no MTP;
- each decode step streams every weight that takes part in the step
  from VRAM exactly once: all transformer-layer weights and `lm_head`.
  The **input embedding table is not streamed**: a step gathers one row
  of it. The vision tower (if any) and an MTP head (when MTP is off) are
  not read;
- KV-cache reads are negligible at short context (checked below);
- compute, kernel-launch and framework overheads are ignored, so the
  result is an upper bound.

```
   ceiling_tok_per_s = peak_bandwidth / bytes_read_per_token
```

### The reference model, Qwen3-8B-NVFP4

Bytes read per token (derived from public parameter counts; see
[`nvfp4-coldstart.md`](nvfp4-coldstart.md) Sec. 2):

- NVFP4 transformer weights: ~6.95 B params x 0.5625 B = **~3.91 GB**
- BF16 `lm_head`: 151 936 x 4 096 x 2 B = **~1.24 GB**
- -> **~5.15 GB per token**

The checkpoint also holds a separate BF16 embedding table of the same
size (vLLM reports 5.98 GiB of weights loaded, which matches body +
two vocabulary matrices), but only one row of it is read per token.

The bench prompts are short (10-50 input tokens, at most 256 output
tokens), so the KV read per step is at most about 306 tokens x
72 KiB = 23 MB, 0.4 % of the weight bytes: negligible.

```
   ceiling = 640 GB/s / 5.15 GB ~ 124 tok/s      (130 tok/s at 672 GB/s)
```

**Measured** (bench `tps_sustained_p50`): the median (type 7) over 40
short prompts of each request's rate, tokens / (time of last token -
time of first token), with tokens counted as characters/4 (Sec. 4),
reasoning tokens included, temperature 0. The bench stores only the
median, so no interval can be computed, and each value is **one run**:

| Run | tok/s | Utilisation at 640 GB/s | at 672 GB/s | Conditions |
|---|---|---|---|---|
| 2026-05-02 | 98.3 | 0.79 | 0.75 | quoted in bench-results.md; context not recorded; not retained |
| 2026-05-05 | 102.1 | 0.82 | 0.78 | `--max-model-len 131072`; run summary log |
| 2026-07-17 | 110.9 | 0.89 | 0.85 | ctx 32768, vLLM v0.22.1 |

So batch-1 decode ran at roughly **75-90 % of the model's ceiling**.
How much of the 13 % spread between these runs is noise? Repeated
single runs of the same models a few days apart differed by -1.6 % to
+3.8 % (eight models, 2026-05-02 vs 05-05; the +3.8 % is this model; SD
of the log ratio 1.8 %), which corresponds to a single-run coefficient
of variation of about 1.3 % (95 % CI roughly 0.8-2.6 %, chi-square,
assuming normal log ratios; `scripts/stats/perf_bench_cache.py`,
[statistics-primer.md](statistics-primer.md) Sec. 9, "the unit of
replication"). That comparison is itself rough: the 2026-05-02 values
are quoted, not retained, and the contexts may have differed. The spread here is
well above that, but the image and the context also changed between
these runs, so it cannot be attributed to any one cause (primer
Sec. 4.3). Where the remaining 10-25 % of the bandwidth goes -- NVFP4
dequantisation, kernel launch latency, attention and RMSNorm compute,
sampling, framework overhead -- has not been measured here; those are
candidate causes, not findings.

### A better test: two builds of one model

The ceiling formula predicts that, other things equal, decode speed
is inversely proportional to the bytes read per token. The retained
data contain a direct check. Two Qwen3.8-27B builds differ mainly in
their weight format; from their safetensors headers (layer weights
plus `lm_head`; embedding table, vision tower and MTP head excluded):

| Build | Bytes read per token | Decode rate, median (95 % CI) | Utilisation at 640 / 672 GB/s |
|---|---|---|---|
| W4A16 AutoRound (`-devai`) | 13.89 GB | 38.9 tok/s (38.7-39.3) | 0.84 / 0.80 |
| NVFP4 (`-devai`) | 14.99 GB | 36.4 tok/s (36.2-36.6) | 0.85 / 0.81 |

Conditions: 2026-09-22, same day, vllm-devai image
`sha256:0a94e7982b85...`, MTP off, temperature 0, **engine-counted**
tokens from vLLM's per-request log, rate = tokens / (engine elapsed -
the run's client-side median time to first token). Both rows use the
same 34 prompts of the bench's 40 (the first prompt excluded, and
prompts with fewer than 64 output tokens in either run). Each build ran
**once**, so the order-statistic intervals (primer Sec. 8) cover
prompt-to-prompt variation only, not run-to-run variation.

The bytes predict a speed ratio of 14.99 / 13.89 = **1.080**. Paired
over the 34 prompts, the observed ratio is **1.069** (geometric mean of
per-prompt ratios; 95 % bootstrap CI over prompts 1.067-1.071; the ratio
of medians over the 21-22 prompts with at least 128 output tokens is
1.067). At face value the pure bytes model is rejected (1.080 lies
outside the interval), but the interval does not include run-to-run
noise, which at about 1-2 % per run is larger than the 1 % shortfall.
A refined model that lets only the bandwidth-bound share of each step
scale with bytes -- about 85 % here, the mean of the two utilisations in
the table -- predicts 1 + 0.080 x 0.849 = **1.068**, in line with the
observation. The test is not perfectly
clean -- the two formats also use different GEMM kernels -- but it
compares the same base model, prompts, day and image, which no
cross-model comparison can (`scripts/stats/perf_ceiling.py`; the
per-build medians and intervals are in `perf_engine_runs.py`,
`build_paired_AR_over_NV.mtp_off`).

### The BF16 case -- and a warning about bounds

The R1-Distill models in [`bench-results.md`](bench-results.md) ship in
**BF16**. For DeepSeek-R1-Distill-Llama-8B (8.03 B params, separate
embedding and `lm_head` matrices of 128 256 x 4 096):

```
   bytes_read_per_token = (8.03 B params x 2 B) - 1.05 GB embedding table ~ 15.0 GB
   ceiling              = 640 / 15.0 ~ 42.6 tok/s        (44.8 at 672 GB/s)
   measured             = 42.51 and 42.48 tok/s          (two single runs,
                                                          2026-05-02 and 2026-05-05,
                                                          characters/4)
```

That is 99.7 % of the 640 GB/s ceiling, higher than any other model
here (0.77-0.89). A measurement at or above a physical bound is a sign
of **bias**, not of perfect efficiency. Likely suspects: the
characters/4 token estimate (if the Llama-3 tokenizer averages more
than 4 characters per token on this text, the count is inflated) and
an understated peak bandwidth. It is not evidence for or against the
bandwidth model; the two-build test above is.

Across models, Qwen3-8B-NVFP4 decoded 2.3-2.6x faster than
R1-Distill-Llama-8B (BF16) in these single runs, against a byte ratio
of 15.0 / 5.15 = 2.9x. These are different models with different
tokenizers, so the comparison illustrates the model rather than testing
it (primer Sec. 4.3).

### Why long contexts slow decode further

At long context, the KV cache is no longer negligible. A worked example
with Qwen3-8B-NVFP4's shapes and FP8 KV, for a sequence that has
reached 128 K tokens -- **hypothetical**: this checkpoint serves at most
32 K as delivered (position limit 40 960):

```
   weights + KV = 5.15 GB + 9.66 GB ~ 14.8 GB per token
   ceiling     = 640 / 14.8 ~ 43 tok/s   (down from 124 tok/s at ~0 KV)
```

Once KV bytes dominate, the ceiling becomes roughly **inversely
proportional** to the context length (double the context, halve the
ceiling). This is why a model "feels faster" at the start of a
conversation and "feels slower" deep into a long session. The only depth
measurements here are for the Qwen3.8-27B builds with MTP on, where only
16 of 64 layers keep a KV cache: client-reported decode went from about 104 tok/s at
3.0 K prompt tokens to 88-91 tok/s at 68.7 K (one request per depth,
2026-09-21). That is consistent in direction, but it is not a test of
the formula.

---

## 8. TTFT math -- what makes the first token slow

`ttft_ms_first` (in the bench cache) and steady TTFT are two very
different things.

### Cold-start TTFT -- about 45 s for Qwen3-8B-NVFP4

The first request to a freshly recreated container pays the full
cost of phases 1 - 11 in
[`nvfp4-coldstart.md`](nvfp4-coldstart.md). For this model the bench
recorded 45.6 s (one request, 2026-05-02; a quoted value, not retained),
then 43.5 s (2026-05-05) and 56.0 s (2026-07-17, ctx 32768): one run
each. The router's launch-to-ready time, which excludes the first request
itself, had a median of 44.5 s over 16 launches on the 2026-04/05 image
(range 42-54 s) and 55-67 s over 3 launches on the 2026-07 one.

vLLM logs the start-up phases itself, so they can be separated. For
the three launches of this model in 2026-07 (ctx 32768;
`perf_phases.py`):

- Container start + Python imports: 7-18 s
- API server and engine-core start (config, tokenizer, process spawn):
  14 s
- CUDA context init and model loading, including the weight copy:
  2-3 s (weights 1.5-1.6 s for 5.98 GiB; PCIe alone would allow
  ~0.1 s for 6 GB, so the copy is not PCIe-bound, but it is small
  either way)
- vLLM's "init engine" (profiling run, torch.compile, kernel JIT and
  autotune, KV-pool allocation, CUDA-graph capture): 30 s, of which
  compilation 15 s and graph capture 1 s; the rest is not broken down
  by the log
- The router's next `/health` poll (every 2 s) sees the engine ready:
  2 s. These rows add up to the 55-67 s launch-to-ready time.
- Then the first prefill and first decode step of the actual prompt

vLLM's init phase is the largest single phase, but compilation is only
half of it here, and container start plus imports (7-18 s) take about as
long as compilation (15 s). For the 27B builds init is 96-236 s
depending on which caches are warm, of which compilation is 40-47 s; the
rest is not attributed by the log (`nvfp4-coldstart.md` Sec. 1).

### Steady-state TTFT -- about 33 ms p50 for Qwen3-8B-NVFP4

Measured: median time to first token over the 39 warm prompts of one
bench run, client-side through the router (so it includes HTTP and
proxy time): 32.7 ms (p95 34.5 ms, 2026-05-02, quoted value, not
retained), 32.3 ms (2026-05-05) and 33.0 ms (2026-07-17). Quantiles are
type 7 (linear interpolation, statistics-primer.md Sec. 8); no interval
exists because only the summaries were stored.

Why so short? For the bench's short prompts (~10-50 tokens):

- Prefill of ~50 tokens: the matrix work for so few tokens is small
  next to reading the weights once, so this prefill costs roughly one
  full weight read, like a decode step: ~8 ms at 640 GB/s and 5.15 GB
  (an estimate; the prompt length at which prefill becomes
  compute-bound depends on the card's FP4 throughput, which was not
  measured here).
- First decode step: another weight read, ~8 ms; plus attention compute
  and sampling.

So about 16 ms of GPU time plus router and HTTP overhead: the measured
~33 ms is of the order this arithmetic suggests. It is a plausibility
check, not a decomposition (the router and HTTP time were not measured
separately).

### TTFT scales with prompt length -- and why NVFP4 prefills faster

Long prompts make prefill expensive. For a 10 K-token prompt:

- Prefill: sequence length 10 K through every Linear -> seconds of
  tensor-core work (compute-bound now).
- First decode: same ~8 ms as before.

TTFT for long prompts grows roughly as **prompt_tokens x a
per-token cost**. The bench measures short prompts only; the long-prompt
data here come from one ad-hoc run per build on 2026-09-21 (Qwen3.8-27B,
MTP on, temperature 0, one request per prompt size; client-side TTFT;
engine-counted prompt tokens):

| Prompt tokens | TTFT, W4A16 AutoRound | TTFT, NVFP4 | Ratio | Prefix-cache reuse |
|---|---|---|---|---|
| 2 958 | 1.84 s | 0.71 s | 2.58 | none (first long prompt) |
| 7 857 | 3.96 s | 1.50 s | 2.64 | partial |
| 32 383 | 18.66 s | 8.51 s | 2.19 | partial |
| 68 663 | 34.36 s | 19.29 s | 1.78 | partial |

The prompts were built for nominal sizes of 3 932 - 91 750 tokens by a
builder that assumes 3.5 characters per token (`_CHARS_PER_TOKEN` in
`scripts/bench/bench_longctx.py`) and came out 25 % smaller when the
engine counted them -- an example of why "what counts as a token"
matters (Sec. 4). Two more
caveats apply:

- **Prefix-cache reuse.** Each prompt extends the previous one with the
  same filler text, so vLLM's prefix cache served part of each later
  prompt: its cumulative hit rate (logged every 10 s) was 18.7 % after
  the first three prompts and 34.6 % after the fourth, identically in
  both runs. Only the first row is a clean prefill measurement: about
  1 600 (W4A16) and 4 150 (NVFP4) prompt tokens/s. For the later rows,
  "prompt tokens / TTFT" overstates the compute rate by an unknown
  amount per request; if the implied cache hits are attributed as the
  hit rates suggest, the rate on the newly computed tokens was about
  1 100-1 400 (W4A16) and 1 960-3 190 (NVFP4) tokens/s
  (`scripts/stats/perf_engine_runs.py`, `doc_figures.depth_prefix_cache`).
  Since 2026-09-27 the builder opens every prompt with a salt unique to
  the request (`_build_long_prompt(..., salt=...)`), so a repeat of this
  measurement shares no prefix across requests beyond what the chat
  template puts first, and the long-context probe records the engine's
  own prompt-token count (`input_tokens`).
- **The ratio between builds is less affected**, because both runs
  received the same prompts in the same order and reused the prefix
  cache identically. It is 2.58 on the clean first prompt and 2.64,
  2.19 and 1.78 on the later ones (not monotone in length; one request
  each).

This pair of builds is a good lesson in the two phases. Their decode
rates are within 7 % of each other, as their bytes per token (Sec. 7)
predict, because decode is bandwidth-bound. Their prefill differs by
2.6x on the clean prompt, because prefill is compute-bound, and the
NVFP4 build computes with Blackwell's native FP4 tensor cores while the
W4A16 build expands its int4 weights to 16 bits for each matmul. The 8B
reference was not measured at long prompts; the rule of thumb of
0.1 - 1 ms per prompt token for an 8 B model is external.

---

## 9. Putting it all together -- the headline table

Qwen3-8B-NVFP4 on the RTX PRO 4000 Blackwell. "One run" means one
bench run of 40 short prompts; ranges are across the dated single runs
in Sec. 7-8.

| Metric | Value | Kind, n, date |
|---|---|---|
| Tokeniser vocabulary | 151 936 tokens | `config.json` `vocab_size` |
| Cold-start TTFT (launch + first request) | 45.6 s; 43.5 and 56.0 s in later runs | measured, 1 request per run, 2026-05-02 (quoted, not retained) / 05-05 / 07-17 |
| Router launch-to-ready | median 44.5 s (range 42-54 s); 55-67 s in 2026-07 | measured, 16 launches (2026-04-29..05-05) and 3 launches (2026-07) |
| Steady-state TTFT p50 / p95 | 32.7 / 34.5 ms (2026-05-02, quoted, not retained) | measured, 39 requests, one run; type-7 quantiles; later p50s 32.3 and 33.0 ms |
| Decode rate, batch 1, short context | 98.3, 102.1, 110.9 tok/s | measured, median of 40 requests, one run each, characters/4 token count |
| Decode ceiling | ~124 tok/s (130 at 672 GB/s) | derived: 640 GB/s / 5.15 GB |
| Fraction of the ceiling reached | 0.79-0.89 (0.75-0.85 at 672 GB/s) | derived from the two rows above |
| GPU memory bandwidth | ~640 GB/s | repo figure, no primary source; 672 GB/s commonly published |
| Peak observed VRAM | 22.53 GiB (2026-05-02, quoted, not retained); 22.31 GiB (2026-07-17) | measured, 1 Hz device-wide samples, one run each; see `nvfp4-coldstart.md` Sec. 2 |

For other models on the same card (bytes per token from public
parameter counts or safetensors headers; ceilings at 640 GB/s):

| Model | Measured tok/s (runs) | Bytes per token | Ceiling | Fraction of ceiling |
|---|---|---|---|---|
| Qwen3-8B-NVFP4 | 98.3, 102.1, 110.9 (3 single runs, chars/4) | 5.15 GB | ~124 | 0.79-0.89 |
| Qwen3.5-9B-NVFP4 | 55.7, 55.3, 57.6 (3 single runs, chars/4) | 8.93 GB | ~72 | 0.77-0.80 |
| R1-Distill-Llama-8B (BF16) | 42.5, 42.5 (2 single runs, chars/4) | 15.0 GB | ~43 | ~1.00 -- suspect, see Sec. 7 |
| Qwen3.8-27B W4A16 AutoRound | 38.9 (one run, engine-counted, 34 prompts) | 13.89 GB | ~46 | 0.84 |
| Qwen3.8-27B NVFP4 | 36.4 (one run, engine-counted, 34 prompts) | 14.99 GB | ~43 | 0.85 |

Qwen3.5-9B's 248 K-token BF16 `lm_head` alone is 2.03 GB, so it reads
8.93 GB per token and sits in the same band as the dense models. (Its
linear-attention layers are gated DeltaNet, not Mamba.)

Quantisation lowers the bytes read per token and so raises the
ceiling; batch-1 decode stays bandwidth-bound.

---

## 10. Practical implications

- **Choose models by your dominant phase.** Interactive agents care
  about TTFT (prefill cost + first decode) and short-context decode
  rate. Batch summarisers care about sustained tok/s on long context.
- **Use the smallest weight format your quality bar tolerates.**
  Fewer bytes per token means a higher decode ceiling (Sec. 7), and a
  format with native tensor-core support also speeds up prefill
  (Sec. 8). In single runs on this card NVFP4 8B decoded 2.3-2.6x
  faster than a BF16 8B model (different models, so the comparison is
  confounded). For quality, compare the dated scores and their
  intervals in [`bench-results.md`](bench-results.md); do not assume
  either a quality cost or its absence.
- **Watch context length.** Once KV bytes dominate the per-token read,
  the decode ceiling is roughly inversely proportional to the context
  (model-based, Sec. 7). The
  [`nvfp4-coldstart.md`](nvfp4-coldstart.md) Sec. 2 diagram shows the
  KV column growing relative to weights.
- **Cold start is a one-time cost -- but it is only "one time" per
  `(model, ctx, MTP setting)` combination.** Changing any of them
  re-walks phases 1-11 (router launch-to-ready medians: 44.5 s for the
  8B reference on the 2026-04/05 image, 125 s for the 27B NVFP4 build;
  `nvfp4-coldstart.md` Sec. 1); the reasoning override (`::nothink`
  etc.) is a per-request rewrite and costs nothing. The router exposes
  `@<ctx>` at picker time so users make the choice deliberately.
- **PCIe is not the bottleneck for (re)loads.** Loading the 6 GB
  Qwen3-8B-NVFP4 weights took 1.5-1.6 s (vLLM log, 3 launches); the
  whole launch took 55-67 s, of which vLLM's init phase (compilation,
  profiling, graph capture) was 30 s. If your cold start is slow, the
  answer is almost never "buy more PCIe lanes."
- **Tokeniser inefficiency is a real cost.** A non-English prompt
  may use 3 - 5x more tokens than the same content in English
  (external rule of thumb).
  That multiplies prefill cost, KV cache size, and per-reply token
  count.

---

## 11. References

### Tokenisation (the BPE family)

- Sennrich, R., Haddow, B., Birch, A. (2016). *Neural Machine
  Translation of Rare Words with Subword Units.* ACL 2016.
  [arXiv:1508.07909](https://arxiv.org/abs/1508.07909). Original
  paper that adapted BPE to NLP.
- Kudo, T., Richardson, J. (2018). *SentencePiece: A simple and
  language independent subword tokenizer and detokenizer for Neural
  Text Processing.* EMNLP 2018.
  [arXiv:1808.06226](https://arxiv.org/abs/1808.06226). Reference
  implementation used in T5, mBART, etc.
- Karpathy, A. (2024). *Let's build the GPT Tokenizer.* YouTube,
  2 h 13 min. The single best educational walkthrough of BPE in
  practice; builds a working tokeniser from scratch.
- HuggingFace tokenizers library docs:
  <https://huggingface.co/docs/tokenizers> -- actual source code for
  GPT-, Llama-, and Qwen-style BPE tokenisers.
- OpenAI tokeniser FAQ (rule-of-thumb 4 chars / token):
  <https://help.openai.com/en/articles/4936856>.

### Inference performance (prefill, decode, KV cache)

- Williams, S., Waterman, A., Patterson, D. (2009). *Roofline: An
  insightful visual performance model for multicore architectures.*
  CACM 52(4). Origin of the compute-bound vs memory-bound distinction
  used in Sec. 5 - Sec. 7.
- Kwon, W. *et al.* (2023). *Efficient Memory Management for Large
  Language Model Serving with PagedAttention.* SOSP 2023.
  [arXiv:2309.06180](https://arxiv.org/abs/2309.06180). The vLLM
  paper; explains paged KV, prefill vs decode batching, and the
  scheduler that ships in the `vllm/vllm-openai` image this project
  uses.
- Pope, R. *et al.* (2022). *Efficiently Scaling Transformer
  Inference.* MLSys 2023.
  [arXiv:2211.05102](https://arxiv.org/abs/2211.05102). Provides the
  "decode is bandwidth-bound" framing for batch size 1 and shows the
  arithmetic-intensity rooflines for various model sizes.
- vLLM documentation on prefill / decode separation and chunked
  prefill: <https://docs.vllm.ai/en/latest/usage/engine_args.html>
  and <https://docs.vllm.ai/en/latest/design/v1/prefix_caching.html>.

### Hardware bandwidths

- NVIDIA Blackwell architecture whitepaper (2024 / 2025) -- HBM3e and
  GDDR7 bandwidth figures.
- NVIDIA RTX PRO 4000 Blackwell product page: 24 GB GDDR7,
  <https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/>.
  The card's peak memory bandwidth should be taken from its datasheet
  (or computed from `nvidia-smi -q` bus width and memory clock); this
  repo's 640 GB/s has no recorded source (Sec. 6).
- JEDEC GDDR7 specification (JESD239-1, 2024): 28 - 32 Gbps per pin
  signalling rate.
- PCI-SIG PCI Express Base Specification 5.0: 32 GT/s per lane -> 64
  GB/s peak for x16 (about 55 GB/s practical after framing overhead).

### Project-internal

- [`bench-results.md`](bench-results.md) -- leaderboard, methodology,
  TPS counting fix, NVFP4 vs BF16 comparison (dated snapshots; each
  table is one run per model).
- [statistics-primer.md](statistics-primer.md) -- Sec. 1 (single runs),
  Sec. 4.3 (confounding), Sec. 8 (quantile definitions, the median
  interval), Sec. 9 (per-request vs aggregate rates, what counts as a
  token, ratios and the bootstrap, drift and replication, model-based
  ceilings), Sec. 14 (reporting layout).
- Reproduction: `scripts/stats/perf_bench_cache.py` (bench-cache TPS,
  TTFT and VRAM fields, run-to-run spread), `perf_ceiling.py` (bytes
  per token from safetensors headers, ceilings, the two-build test),
  `perf_engine_runs.py` (engine-counted per-request rates, depth and
  prefill runs, the characters/4 check), `perf_coldstart.py` (router
  launch-to-ready), `perf_phases.py` (vLLM start-up phases and memory
  lines, the 2026-05-05 bench-run summaries), `perf_client_logs.py`
  (the concurrency tool's characters per token). Values dated
  2026-05-02 survive only as quoted in `bench-results.md` and earlier
  versions of these pages.
- [`nvfp4-coldstart.md`](nvfp4-coldstart.md) -- graphviz timeline of
  the 11 cold-start phases plus the per-component VRAM budget.
- [`nvfp4-number-formats.md`](nvfp4-number-formats.md) -- beginner's
  guide to NVFP4, FP8, BF16 and the rest of the number-format
  ecosystem.
- [`router.md`](router.md) -- request rewrite chain, including how
  the picker's `@<ctx>` suffix becomes a vLLM / SGLang launch flag and
  `::reasoning` a per-request body rewrite.
- `deploy/.bench-cache.json` -- raw rolled-up cache row consumed by
  the picker.
- `/var/cache/devai/bench/inspect-logs/*.eval` -- full per-sample
  inspect_ai logs for the GSM8K / HumanEval / tools_use tasks.

---

## Changes to this page (2026-09-27)

Earlier versions of this page stated, and this version corrects:

- tied embeddings and 5.1 GB read per token for Qwen3-8B-NVFP4 (the
  checkpoint has two vocabulary matrices, only `lm_head` is streamed:
  5.15 GB, ceiling ~124 tok/s), and one run's 78 % utilisation as the
  result (three single runs: 0.79-0.89), with the remainder attributed
  to specific causes that were never measured;
- the BF16 R1-Distill case as "proof" that decode is bandwidth-bound
  (it used the total including the unread embedding table, and a
  measurement at 99.7 % of a bound points to bias; the two-build test in
  Sec. 7 is the evidence), and a 2.3x NVFP4/BF16 gap "exactly" as
  predicted (the byte ratio is 2.9x, and the models differ);
- Qwen3.5-9B with a ~110 tok/s ceiling and ~50 % utilisation blamed on
  "hybrid Mamba" layers (8.93 GB per token, ceiling ~72 tok/s,
  utilisation 0.77-0.80; the layers are gated DeltaNet);
- a 128 K KV cache of 9.4 GB for Qwen3-8B (9.66 GB with 1 K = 1 024
  tokens; hypothetical, the checkpoint serves at most 32 K) and a
  ceiling that falls "proportionally" with context (inversely
  proportional, once KV dominates);
- "no per-phase instrumentation" for cold start, CUDA graph capture as
  the longest pole (1 s; vLLM's init phase with compilation is), and a
  0.1 s PCIe weight copy (vLLM logs 1.5-1.6 s);
- a short-prompt prefill of "a few milliseconds" of compute-bound work
  (it costs about one weight read), and "no long-prompt curve" (the
  2026-09-21 depth runs exist, with a prefix-cache confound);
- tokens-per-second values without saying they count characters/4, a
  640 GB/s bandwidth attributed to the datasheet (no source is
  recorded), and a peak VRAM in GB that is GiB;
- NVFP4 "doubling" decode rate with quality "within the noise floor"
  (quality claim dropped; see bench-results.md), and a recreate per
  `(model, ctx, reasoning override)` (the reasoning override is a
  per-request rewrite; an MTP change does recreate).
