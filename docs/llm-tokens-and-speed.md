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

All concrete numbers come from this project's own bench harness
applied to **`nvidia/Qwen3-8B-NVFP4`** running on a **NVIDIA RTX 4000
PRO Blackwell (24 GB GDDR7, ~640 GB/s)** workstation card. The
post-fix numbers are the ones in
[`bench-results.md`](bench-results.md); please prefer that file when
the rolled-up cache row in `deploy/.bench-cache.json` and this doc
disagree.

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

Useful rules of thumb for English text on Latin-vocab tokenisers:

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
amortised over many parallel ops on the same weights. This phase is
**compute-bound** on modern GPUs.

Prefill speed typically lands in the *thousands of tokens per second*
for small-to-medium models on a Blackwell card.

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
cores are mostly idle -- they could process 100x more matrix work,
but they have to wait for the next chunk of weights to arrive from
VRAM. This is the central performance reality of single-stream LLM
serving.

Decode speed for the reference model is **98.3 tok/s** in this
project's bench (Qwen3-8B-NVFP4, post-fix; see Sec. 7 for the math).

### 5.3 Why this matters operationally

- **TTFT** (time-to-first-token) is dominated by prefill: long
  prompt -> long prefill -> user waits longer for the first token.
- **Sustained tok/s** is the decode rate: how fast the reply streams
  *after* the first token. Long generations spend most of their wall
  time here.
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

The reference card -- **NVIDIA RTX 4000 PRO Blackwell** -- has 24 GB
of GDDR7 with a documented **~640 GB/s memory bandwidth**
(see [`bench-results.md`](bench-results.md), top of file).

### What each tier matters for

| Tier | When it matters | Example for Qwen3-8B-NVFP4 |
|---|---|---|
| L1/L2 cache | Tiny tensors that fit on-chip -- irrelevant to weights | rare for inference |
| HBM3e / GDDR7 (VRAM) | **Decode bandwidth**: every generated token requires reading all weights | binds 98.3 tok/s steady decode |
| PCIe Gen5 | Moving the model from system RAM to VRAM during cold start | ~0.1 s for the 6 GB checkpoint |
| DDR5 RAM | Holding the safetensors mmap before / during page-cache warm-up | ~50 GB/s, rarely the bottleneck |
| SSD / NVMe | First-time load from disk; cold page cache | ~12 GB/s on Gen5 SSD |
| Network | Streaming reply tokens to the client | trivial for text |

The gap between VRAM bandwidth and PCIe bandwidth is roughly
**10 x**. The gap between VRAM and L2 is another **10 x**. This
hierarchy is exactly why a model that lives in VRAM serves at tens
of tok/s, the same model offloaded to system RAM serves at
sub-1 tok/s, and the same model served from disk-paged-in is
unusable.

---

## 7. Decode speed math -- why Qwen3-8B-NVFP4 hits ~98 tok/s

Now we can connect the dots. For each generated token, decode at
batch=1 must read approximately:

```
   bytes_per_token = (model weights on device) + (KV cache up to current position)
```

For Qwen3-8B-NVFP4 (see [`nvfp4-coldstart.md`](nvfp4-coldstart.md)
Sec. 2 for the breakdown):

- NVFP4 transformer weights: **~3.9 GB**
- BF16 embeddings + lm_head (tied): **~1.2 GB**
- -> live weights: **~5.1 GB**

The bench prompts are short (10 - 50 input tokens, generating up to
256 output tokens -- see `scripts/bench/data/latency_prompts.jsonl`),
so the KV-cache term is small (under 100 MB) and we can ignore it.

**Theoretical decode ceiling** (perfect memory bandwidth utilisation,
no compute overhead):

```
   ceiling_tok_per_s = bandwidth / weights_per_token
                     = 640 GB/s / 5.1 GB
                     ~ 125 tok/s
```

**Measured** (`tps_sustained_p50` in `bench-results.md`):

```
   measured = 98.3 tok/s
   utilisation = 98.3 / 125 ~ 78 % of peak memory bandwidth
```

That 78 % is excellent for batch-size-1 decode. The remaining ~22 %
goes to NVFP4 dequantisation cost (per-block FP8 scales applied
before each matmul), framework overhead, kernel launch latency, and
attention/RMSNorm compute.

### The same math for BF16 -- proves it's bandwidth-bound

The R1-Distill family in [`bench-results.md`](bench-results.md) ships
in **BF16** rather than NVFP4. For an 8 B BF16 model:

```
   bf16_weights = ~16 GB on device
   ceiling      = 640 GB/s / 16 GB ~ 40 tok/s
   measured     = 42 tok/s for DeepSeek-R1-Distill-Llama-8B
```

The measured number is *at the theoretical ceiling*. There is
literally no way to make BF16 batch=1 decode faster on this card
without changing either the bandwidth (different GPU) or the
per-token byte budget (smaller weights -> quantisation).

This is the cleanest possible empirical proof that batch=1 LLM decode
is memory-bandwidth-bound. Both data points (NVFP4 at 78 % of its
theoretical ceiling; BF16 at 100+ % of its theoretical ceiling within
measurement noise) trace back to the same `bandwidth / weight_size`
formula. NVFP4 wins not because tensor cores are faster (although
they are) but because each generated token only has to *move* a
quarter of the bytes a BF16 model would move.

> **From `bench-results.md`'s "Architectural finding" section:** the
> ~2.3x tok/s gap between NVFP4 and BF16 on this card is exactly
> what you would predict from `(bf16 bytes / nvfp4 bytes)` after
> accounting for the BF16 case running closer to 100 % of bandwidth
> peak than NVFP4 does.

### Why long contexts slow decode further

At long context, the KV cache is no longer negligible. For
Qwen3-8B-NVFP4 with FP8 KV at 128 K context:

```
   weights + KV = 5.1 GB + 9.4 GB = 14.5 GB per token
   ceiling     = 640 / 14.5 ~ 44 tok/s   (down from 125 tok/s at 0 KV)
```

Long context cuts the bandwidth ceiling proportionally. This is why
the same model "feels faster" at the start of a conversation and
"feels slower" deep into a long session -- the KV cache grew and the
per-token byte budget grew with it.

---

## 8. TTFT math -- what makes the first token slow

`ttft_ms_first` (in the bench cache) and steady TTFT are two very
different things.

### Cold-start TTFT -- 45.6 s for Qwen3-8B-NVFP4

The first request to a freshly recreated container pays the full
cost of phases 1 - 11 in
[`nvfp4-coldstart.md`](nvfp4-coldstart.md). The dominant components
on this card are:

- Container start + Python imports (a few seconds)
- Weight H2D copy over PCIe (~0.1 s for 6 GB at PCIe Gen5)
- CUDA context init + cuBLAS workspaces
- **CUDA graph capture** with NVFP4 cutlass JIT autotune -- usually
  the longest pole
- First prefill of the actual prompt
- First decode step

The 45.6 s figure measured here lumps all of those together -- the
project does not yet have per-phase instrumentation.

### Steady-state TTFT -- 32.7 ms p50 for Qwen3-8B-NVFP4

Once warm, TTFT is dominated by **prefill of the prompt + first
decode step**. For the bench's short prompts (~10-50 tokens):

- Prefill ~50 tokens through an 8 B NVFP4 model on Blackwell tensor
  cores: a few milliseconds. Compute-bound, but these are small
  matmuls so total work is tiny.
- First decode step: read all weights once -> at 640 GB/s and 5.1 GB,
  about 8 ms; plus attention compute and sampling.

32.7 ms total is consistent with that arithmetic and matches the
expectation that prefill is fast for short prompts on a modern card.

### TTFT scales with prompt length

Long prompts make prefill expensive. For a 10 K-token prompt:

- Prefill: sequence length 10 K through every Linear -> seconds of
  tensor-core work.
- First decode: same ~8 ms as before.

Empirically, TTFT for very long prompts on Blackwell-class cards is
on the order of **prompt_tokens x small_constant** -- usually
0.1 - 1 ms per prompt token for an 8 B model, much higher for 70 B
class. The bench harness only measures short-prompt TTFT, so we don't
have a long-prompt curve here; if you need it, the
`make bench-vllm` Makefile target can be re-run with custom
`scripts/bench/data/latency_prompts.jsonl` content.

---

## 9. Putting it all together -- the headline table

Cross-referencing [`bench-results.md`](bench-results.md) for the
post-TPS-fix numbers on this card:

| Metric (Qwen3-8B-NVFP4 on RTX 4000 PRO Blackwell) | Value | Source |
|---|---|---|
| Tokeniser vocabulary | 151 936 tokens | `config.json` `vocab_size` |
| Cold-start TTFT (recreate + load + capture + prefill + first decode) | **45 600 ms** | bench `ttft_ms_first` |
| Steady-state TTFT (warm; ~10-50 token prompt) | **32.7 ms p50 / 34.5 ms p95** | bench `ttft_ms_steady_p50/p95` |
| Sustained decode throughput (batch=1, ~zero KV) | **98.3 tok/s** | bench `tps_sustained_p50` (post-fix) |
| Theoretical decode ceiling (bandwidth-bound, batch=1) | ~125 tok/s | (640 GB/s) / (5.1 GB) |
| Bandwidth utilisation in decode | ~78 % | measured / ceiling |
| GPU memory bandwidth | ~640 GB/s | RTX 4000 PRO Blackwell datasheet |
| Peak observed VRAM | 22.53 GB | bench `peak_vram_gb` |

For a different model class on the same card:

| Quantisation x class | TPS measured | Bandwidth ceiling | Utilisation |
|---|---|---|---|
| NVFP4 8B (Qwen3-8B-NVFP4) | 98.3 tok/s | ~125 | ~78 % |
| NVFP4 9B (Qwen3.5-9B-NVFP4) | 55.7 tok/s | ~110 | ~50 % (hybrid Mamba slows it) |
| BF16 8B (R1-Distill-Llama-8B) | 42.5 tok/s | ~40 | ~106 % (within measurement noise of ceiling) |

The BF16 entry pegs the bandwidth ceiling exactly. Quantisation moves
the ceiling; it does not break the wall.

---

## 10. Practical implications

- **Choose models by your dominant phase.** Interactive agents care
  about TTFT (prefill cost + first decode) and short-context decode
  rate. Batch summarisers care about sustained tok/s on long context.
- **Use the smallest weight format your quality bar tolerates.**
  NVFP4 over BF16 doubled decode rate on this card while leaving
  GSM8K / HumanEval scores within the noise floor for Qwen3-class
  models. See [`bench-results.md`](bench-results.md) for the actual
  quality-vs-speed comparison.
- **Watch context length.** Doubling context cuts decode ceiling
  roughly in half once KV starts to dominate. The
  [`nvfp4-coldstart.md`](nvfp4-coldstart.md) Sec. 2 diagram shows the
  KV column growing visibly relative to weights at 128 K and 256 K.
- **Cold start is a one-time cost -- but it is only "one time" per
  `(model, ctx, reasoning_override)` triple.** Switching among loaded
  options thrashes phases 1-11. The router exposes `@<ctx>` and
  `::reasoning` suffixes at picker time so users make the choice
  deliberately.
- **PCIe is fast enough not to be the bottleneck for warm
  reloads.** The 6 GB Qwen3-8B-NVFP4 weights move from page cache
  into VRAM in ~0.1 s over PCIe Gen5. If your cold start is slow,
  the answer is almost never "buy more PCIe lanes."
- **Tokeniser inefficiency is a real cost.** A non-English prompt
  may use 3 - 5x more tokens than the same content in English.
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
- NVIDIA RTX 4000 PRO Blackwell product page: 24 GB GDDR7,
  <https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/>.
- JEDEC GDDR7 specification (JESD239-1, 2024): 28 - 32 Gbps per pin
  signalling rate.
- PCI-SIG PCI Express Base Specification 5.0: 32 GT/s per lane -> 64
  GB/s peak for x16 (about 55 GB/s practical after framing overhead).

### Project-internal

- [`bench-results.md`](bench-results.md) -- leaderboard, methodology,
  TPS counting fix, NVFP4 vs BF16 comparison. **Source of truth for
  every measured number in this doc**; if a value here disagrees with
  that file, the bench-results page wins.
- [`nvfp4-coldstart.md`](nvfp4-coldstart.md) -- graphviz timeline of
  the 11 cold-start phases plus the per-component VRAM budget.
- [`nvfp4-number-formats.md`](nvfp4-number-formats.md) -- beginner's
  guide to NVFP4, FP8, BF16 and the rest of the number-format
  ecosystem.
- [`router.md`](router.md) -- request rewrite chain, including how
  the picker's `@<ctx>` and `::reasoning` suffixes propagate to vLLM
  / SGLang launch flags.
- `deploy/.bench-cache.json` -- raw rolled-up cache row consumed by
  the picker.
- `/var/cache/devai/bench/inspect-logs/*.eval` -- full per-sample
  inspect_ai logs for the GSM8K / HumanEval / tools_use tasks.
