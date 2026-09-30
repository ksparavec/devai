# Multi-token prediction (MTP) -- faster decode with the same model

This page covers the trick that lets a model emit several tokens per
forward pass instead of one, without changing the distribution of the
model's outputs (Sec. 6). The canonical name on Google's product pages is **multi-token
prediction (MTP)**; the canonical name in the literature is
**speculative decoding**. They are *almost* the same thing -- they
differ in *who proposes the next K tokens* (a separate drafter model
vs. extra heads stitched into the main model), but the verification
math is identical.

Both names show up in vLLM / SGLang / HuggingFace / Ollama flag docs
because both shapes are live in 2026. This doc explains which one
each provider ships, how this project's router serves it, and what it
measured on this project's card: about **2.5x** faster decode for the
Qwen3.8-27B builds on the bench's fixed set of short prompts at
temperature 0 (paired over 34 prompts, one run per arm; Sec. 7.1), at a
cost of 0.60 GiB more weights and a 24 % smaller KV pool on the NVFP4
build (Sec. 9.3).

Numbers in this doc are of three kinds, and each is labelled: figures
**measured** on this project's RTX PRO 4000 Blackwell (with n, number
of runs and date), figures **derived** from a model with stated
assumptions, and **external** figures reported by vendors or papers and
not reproduced here. [statistics-primer.md](statistics-primer.md)
Sec. 9 and 14 explain how the measured ones are reported.

If you have not yet internalised the prefill / decode split,
[`llm-tokens-and-speed.md`](llm-tokens-and-speed.md) Sec. 5-7 is the
prerequisite for understanding *why* this trick exists. If you have
not yet read [`paged-attention-and-vllm-internals.md`](paged-attention-and-vllm-internals.md),
skim Sec. 1 of that doc -- MTP relies on prefill batching the
drafter's proposals through the verifier, which is the same kernel
path as the prefix-caching / continuous-batching machinery already
covered there.

---

## 1. Why decode is slow -- a one-paragraph recap

A modern LLM emits **one token per forward pass**.
[`llm-tokens-and-speed.md`](llm-tokens-and-speed.md) Sec. 7 derives
the ceiling: each decode step reads every weight and every cached KV
slot once, then writes one new token plus its KV row. For
`Qwen3-8B-NVFP4` on this project's RTX PRO 4000 Blackwell card the
derived ceiling is about `640 GB/s / 5.15 GB ~ 124 tok/s` (assuming
the repo's unsourced 640 GB/s peak), and three single bench runs
measured 98.3-110.9 tok/s, roughly 79-89 % of it. There is no way to
push past that ceiling *as long as the model emits one token per
forward pass*. The GPU is not compute-starved; it is memory-bandwidth
starved (the paired two-build test in `llm-tokens-and-speed.md` Sec. 7
supports this).

MTP attacks the problem from a different angle. Instead of trying to
go faster per pass, it produces **K tokens per pass** in the common
case and falls back to one-per-pass when the proposal is wrong.

---

## 2. The speculative-decoding idea

The mechanism, drawn from Leviathan, Kalman & Matias 2022
([arXiv:2211.17192](https://arxiv.org/abs/2211.17192)):

```
    1. A small fast "drafter" model generates K candidate tokens
       autoregressively. Call them d_1, d_2, ..., d_K.
       Cost: K forward passes through a small model.

    2. The big "target" model takes the prompt plus all K candidates
       as a single input and runs ONE forward pass over the entire
       extended sequence. This gives target logits at positions
       prompt+1, prompt+2, ..., prompt+K, prompt+K+1.
       Cost: 1 forward pass through the big model (batched).

    3. Walk the K candidates left-to-right. For each d_i, compare the
       drafter's predicted distribution at position i against the
       target's distribution at position i (the one we just got from
       the big model). Accept d_i if it passes a probabilistic test
       (Sec. 6); otherwise reject d_i and everything after.

    4. After the longest accepted prefix (say tokens 1..j with
       j <= K), use the target's distribution at position j+1 to
       sample one more token. That token is always accepted because
       it comes from the target itself.

    5. Result: between 1 and K+1 new tokens emitted per round, all
       distributed identically to what the target would have sampled
       on its own.
```

The win comes from step 4's guaranteed +1: even if every drafter
proposal is rejected, the target still produces one fresh token. And
when the drafter is *right* (which it usually is on routine code,
common phrases, repeated structure), you get K+1 tokens in roughly
the time of one big-model decode step.

Why is step 2 cheap? Because **K extra tokens through the verifier
cost roughly the same as one token** in decode mode. Decode is
bandwidth-bound; you pay the same weight read per step (~5 GB for an
8B NVFP4 model) whether you process 1 or 16 positions in that step
(small K does not push into compute-bound territory). The drafter's K
serial passes are cheap because the drafter is tiny.

End-to-end: published speedups are about 1.7-3x depending on drafter,
K and content (external, Sec. 4-5). On this card the Qwen3.8-27B builds
measured 2.52x and 2.60x on short prompts (paired, Sec. 7.1).

---

## 3. Two architectures -- external drafter vs. built-in MTP head

The mechanism above is universal. What differs across model
families is *how step 1 produces the K candidate tokens*. Two
shapes dominate in 2026:

### 3.1 External drafter (Gemma 4, EAGLE/EAGLE3, Medusa, draft-model)

A **separate small transformer** is trained alongside the target.
Gemma 4 calls these **"assistant" models**; EAGLE calls them
**"draft heads"**; the generic vLLM term is **draft_model**. The
drafter has its own weights, lives at its own HuggingFace repo, and
must be loaded into VRAM next to the target.

What makes 2026-era drafters efficient is **KV-cache sharing**: the
drafter sees the *target's* hidden activations at the last layer and
reuses the *target's* KV cache for already-emitted tokens. The
drafter therefore does not have to re-encode the prompt -- it just
needs to be small enough to run K times in less time than the target
takes to run once.

### 3.2 Built-in MTP head (DeepSeek-V3, Qwen3.6, NVIDIA Megatron-MTP)

A **few extra transformer modules** are added on top of the target
during pre-training, sharing the embedding table and most weights
with the main stack. After training, the same checkpoint contains
both the "main" prediction path (one token at a time) and the "MTP"
prediction path (K tokens at a time). No separate drafter file. No
extra HuggingFace repo. From `ls -lh`, the only sign is a small
extra block of weights -- roughly 850 MB BF16 for Qwen3.6's
`mtp_num_hidden_layers=1` arrangement on a 27 B target (external). It
is not free at run time, though: the prepared Qwen3.8-27B NVFP4 build
here loaded 0.60 GiB more with MTP on, and its KV pool shrank by 24 %
(measured, Sec. 9.3).

The two shapes are summarised:

| Aspect | External drafter | Built-in MTP head |
|---|---|---|
| Where weights live | separate repo (`-assistant`, `-eagle3`) | same checkpoint as target |
| Extra VRAM cost | drafter weights + drafter KV (sharable) | head weights (in the checkpoint, but loaded only with MTP on) + KV for the head's own layer; measured here: +0.60 GiB weights, KV pool -24 % (Sec. 9.3) |
| Training | independent (distil from target) | jointly trained with target |
| Example checkpoint | `google/gemma-4-26B-A4B-it-assistant` (801 MB) | `Qwen3.6-27B-Text-NVFP4-MTP` (built into 18 GB) |
| vLLM flag shape | `'{"method":"mtp","model":"<repo>","num_speculative_tokens":N}'` | `'{"method":"deepseek_mtp"/"qwen3_5_mtp","num_speculative_tokens":N}'` (no `model` field) |
| SGLang flag shape | `--speculative-algorithm EAGLE --speculative-draft-model-path <path>` | `--speculative-algorithm NEXTN` (same as EAGLE) |

The verification math (Sec. 6) is the same in both shapes. The only
runtime difference is whether the engine loads two `safetensors`
files or one.

---

## 4. Gemma 4 MTP -- the reference example

Google released MTP drafters for the entire Gemma 4 open-model
family on 2026-05-05. Four target / assistant pairs:

| Target | Target file size | Assistant repo | Assistant file size | TP recommended (NVIDIA) |
|---|---|---|---|---|
| `google/gemma-4-E2B-it` | ~5 GB BF16 | `google/gemma-4-E2B-it-assistant` | **150 MB** | 1 |
| `google/gemma-4-E4B-it` | ~15 GB BF16 | `google/gemma-4-E4B-it-assistant` | **152 MB** | 1 |
| `google/gemma-4-26B-A4B-it` (MoE 25 B / 3.8 B active) | ~48 GB BF16 | `google/gemma-4-26B-A4B-it-assistant` | **801 MB** | 2 |
| `google/gemma-4-31B-it` (dense 31 B) | ~58 GB BF16 | `google/gemma-4-31B-it-assistant` | **896 MB** | 2 |

(File sizes are the BF16 safetensors as published on the HuggingFace
tree API on 2026-05-13; the assistants are single-file checkpoints.)

The drafters are described by Google as **4-layer transformers** --
i.e. dramatically smaller than the targets. The 26 B MoE assistant
is the largest at 801 MB; the edge-tier E2B/E4B assistants weigh in
at just 150 MB. The pattern: **the drafter is about 1-3 % of the
target's weight footprint**.

Two extra optimisations are baked in:

1. **Shared KV cache.** The assistant reads from the target's KV
   blocks (same `block_size`, same head layout) rather than
   maintaining its own. This is the single biggest win -- a naive
   speculative-decoding setup with two independent KV caches would
   double the KV bytes per sequence.

2. **Centroids masking** (E2B/E4B only). Gemma 4's vocabulary is
   ~262 K tokens; computing the `lm_head` dot product over the full
   vocab at every drafter step would dominate runtime for a small
   drafter. The E2B/E4B assistants cluster the vocabulary into ~4 K
   centroids and the drafter only scores those centroids -- a ~45x
   reduction in `lm_head` FLOPs. Enabled automatically when the
   assistant checkpoint advertises `use_ordered_embeddings: true`;
   the 26B-A4B and 31B assistants don't use it.

Google's published headline (external, not reproduced here): **up to
3x decoding speedup on NVIDIA RTX PRO 6000**, with bit-exact identity
to the unaccelerated path (a vendor claim; Sec. 6 explains why exact
identity holds in exact arithmetic but is not guaranteed on a GPU). On Apple Silicon at batch 4-8
they report ~2.2x for the 26 B MoE.

The vLLM serve command from Google's reference recipe page
([docs.vllm.ai .../Gemma4.html](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html)):

```
    vllm serve google/gemma-4-31B-it \
        --tensor-parallel-size 2 \
        --max-model-len 8192 \
        --speculative-config '{
            "model": "google/gemma-4-31B-it-assistant",
            "num_speculative_tokens": 4
        }'
```

The schema for `--speculative-config` is a JSON object; common keys:

| Key | Meaning |
|---|---|
| `method` | `"mtp"`, `"deepseek_mtp"`, `"qwen3_5_mtp"`, `"eagle"`, `"eagle3"`, `"medusa"`, `"ngram"`, `"draft_model"` |
| `model` | path/repo of the drafter (omit for built-in MTP heads) |
| `num_speculative_tokens` | K -- how many tokens the drafter proposes per round; 2-8 typical |

For Gemma 4 the `"method"` defaults to MTP-like behaviour when an
assistant checkpoint is supplied; explicit method values are needed
for DeepSeek and Qwen (Sec. 5).

---

## 5. Non-Google MTP/speculative providers

The technique is not Google-specific. The notable other shapes in
2026:

### 5.1 DeepSeek-V3 / V3.2 -- built-in MTP, 671 B total

DeepSeek-V3's technical report
([arXiv:2412.19437](https://arxiv.org/abs/2412.19437)) was the first
public-frontier-scale model to ship MTP as a *built-in head*. The
architecture: **D=4 MTP modules**, each a single transformer block
plus a shared embedding/output head, predicting the next D tokens
in parallel. The same checkpoint serves both as a one-token-per-pass
generator (main path) and a multi-token drafter (MTP path).

The reported acceptance rate on MTP-1 is **>80 %**, yielding
**~1.8x** decode throughput (external, DeepSeek-V3 technical report).
vLLM flag:

```
    --speculative-config '{"method": "deepseek_mtp",
                           "num_speculative_tokens": 1}'
```

V3 is 671 B total / 37 B active -- not even close to fitting on a
24 GB card. Listed here for completeness; the project's catalog
does not include DeepSeek-V3 base.

### 5.2 Qwen3.6 -- built-in MTP at 24 GB-friendly sizes

Qwen3.6 (released 2026-Q1) ships a single MTP module
(**`mtp_num_hidden_layers=1`** in `text_config`, as the Qwen3.8
checkpoints here also record it) that can be applied recursively up to
N speculative steps. Per-position acceptance is reported as ~87 % /
72 % / 61 % for positions 1/2/3 with `num_speculative_tokens=3`
(external). How many tokens a verification step yields on average
depends on what those rates mean, and the source does not say:

- if each is the probability that positions 1 to i are **all**
  accepted (unconditional; this is how vLLM counts), the verifier's
  guaranteed token plus the plain sum gives 1 + 0.87 + 0.72 + 0.61 ~
  3.2 tokens per step;
- if each is the probability of acceptance **given** that the previous
  position was accepted (conditional), it is 1 + 0.87 + 0.87 x 0.72 +
  0.87 x 0.72 x 0.61 ~ 2.88.

For comparison, this project's Qwen3.8-27B builds logged unconditional
per-position rates of about 0.84 / 0.66 / 0.51 (medians over fourteen
10-s windows, two runs; windows ranged 0.75-0.98 / 0.51-0.90 /
0.37-0.83). Pooled over all windows, vLLM accepted 66 % of drafted
tokens, so a verification step yielded 1 + 3 x 0.66 = 2.98 tokens on
average on the short bench prompts (vLLM's own counters; its logged
mean acceptance length had a per-window median of 3.0; Sec. 7.1).

The catch: stock NVFP4 quantization scripts **drop the MTP head**
because `AutoModelForCausalLM.from_pretrained` doesn't load it.
Community quants restore it in BF16. The relevant 24 GB-class
checkpoint:

| Repo | Size | KV-pool headroom @256K |
|---|---|---|
| `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` | 18.3 GB (NVFP4 weights + ~850 MB BF16 MTP head) | fits with `--kv-cache-dtype fp8 --max-num-seqs 2` |

The author's reported speedup (external): **1.74x** on long-form
decode (207 vs 119 tok/s) with `num_speculative_tokens=3`. vLLM flag:

```
    --speculative-config '{"method": "qwen3_5_mtp",
                           "num_speculative_tokens": 3}'
```

(The `qwen3_5_mtp` method name reflects vLLM's parser registry; the
underlying MTP head is Qwen3.6.)

### 5.3 EAGLE / EAGLE3 -- community drafters for arbitrary targets

[EAGLE](https://arxiv.org/abs/2401.15077) (Li et al. 2024) and
EAGLE3 are *training recipes* for community-built drafters that
target any open model. Within four days of Gemma 4's release, a
community member trained an EAGLE3 head for Gemma-4-31B
([`lujangusface/tw-eagle3-gemma4`](https://huggingface.co/blog/lujangusface/tw-eagle3-gemma4))
and reported a **1.72x speedup** (external) -- slightly slower than
Google's official assistant, but trained with a fraction of the
compute.

SGLang flag shape:

```
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path <hf-repo-or-local-path> \
    --speculative-num-steps 1 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 2
```

The `NEXTN` algorithm in SGLang is an alias for `EAGLE` with the
draft tree pruned to a single chain -- the closest analog to
HuggingFace's `assistant_model` arg.

### 5.4 Medusa -- multiple parallel heads (older)

[Medusa](https://arxiv.org/abs/2401.10774) (Cai et al. 2024) attaches
**multiple independent prediction heads** to the base model, each
predicting the next-+i-th token in parallel. The verification step
then needs a tree-search over the cross-product of all heads' top-K
predictions. Effective in 2024 but largely superseded by EAGLE in
2026 because EAGLE's hidden-state-conditioning gives higher
acceptance with fewer drafted tokens.

### 5.5 N-gram / suffix -- no model needed

vLLM and SGLang both support a **stateless** drafter that proposes
the next K tokens by looking up the longest suffix of the current
output in a recent context buffer (suffix decoding) or in an
n-gram trie. Free at runtime; the trade-off is much lower
acceptance on non-repetitive content. Useful for code completion
and structured-output workloads where the same tokens repeat often.

vLLM flag:

```
    --speculative-config '{"method": "ngram",
                           "num_speculative_tokens": 5}'
```

### 5.6 NVIDIA -- first-party MTP and EAGLE3 drafts

NVIDIA does publish first-party MTP / draft-head assets. The catch
for this project is that **none of them fit a 24 GB card today**;
they are aimed at datacenter B200 / H200 / DGX Spark deployments.
Useful to know about anyway, since the project's catalog *does*
include NVIDIA's NVFP4 quantizations of *other* vendors' MTP models
(the Gemma 4 row above), and NVIDIA's training recipes are what
makes EAGLE3 drafters work.

The NVIDIA-published, MTP-relevant artifacts:

| Repo | Class | Size on disk | What it is | Fits 24 GB? |
|---|---|---|---|---|
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | built-in MTP, MoE | **67 GB** | NVIDIA's flagship open hybrid Mamba-Transformer MoE; ships MTP via shared-weight prediction heads | no |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8` | built-in MTP, MoE | ~120 GB | FP8 variant of the same | no |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` | built-in MTP, MoE | ~240 GB | BF16 variant | no |
| `nvidia/Llama-3.3-70B-Instruct-Eagle3` | EAGLE3 draft head | (small) | 3.2 B EAGLE3 draft head trained by NVIDIA for `meta-llama/Llama-3.3-70B-Instruct`; tightly coupled to the 70B target's hidden state | drafter yes; target ~40 GB at NVFP4 -- no |
| `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` | **no MTP** | 21 GB | Same family but Nano tier; MTP heads were reserved for the Super-120B variant | yes (but no MTP) |
| `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | **no MTP** | ~18 GB BF16 | Hybrid Mamba2-Transformer; speculative decoding not baked in | yes (but no MTP) |

**Nemotron 3 Super** is genuinely interesting on paper: it ships
MTP with the highest reported acceptance length (3.45 mean
accepted tokens, beating DeepSeek-R1 in NVIDIA's own benchmarks;
external)
and a shared-weight head design that stays stable at longer draft
lengths. vLLM flag shape (from NVIDIA's deployment cookbook):

```
    --speculative-config '{"method": "mtp",
                           "num_speculative_tokens": 3,
                           "moe_backend": "triton"}'
```

The TensorRT-LLM equivalent uses `decoding_type: MTP` with
`num_nextn_predict_layers: 3`. But at 67 GB NVFP4 / 120 GB FP8,
the target alone overflows a 24 GB card by 3-5x. This becomes
relevant if/when the project ever runs on B200 hardware.

**Llama-3.3-70B-Instruct-Eagle3** is the only NVIDIA-published
artifact that *would* be drop-in-pairable with the project's
router if the target were small enough -- but EAGLE3 drafters are
trained against a specific target's hidden states and cannot be
swapped onto a smaller Llama 3.x. NVIDIA has not (yet) published
an EAGLE3 head for `Llama-3.1-Nemotron-Nano-8B-v1` or the 9B-v2
hybrid, both of which would fit the project's hardware
comfortably. The PayPal commerce-agent paper
([arXiv:2604.19767](https://arxiv.org/abs/2604.19767)) trained
their *own* EAGLE3 against `Llama-3.1-Nemotron-Nano-8B-v1` and
report 22-49 % throughput gain at gamma=3 (external) -- but that draft head
is not publicly released as of 2026-05.

**Net for this project:** the NVIDIA-branded fit-the-card MTP
path today goes through *NVIDIA's NVFP4 quantization of Google's
Gemma 4* (`nvidia/Gemma-4-26B-A4B-NVFP4`, already in the catalog
and downloaded in 2026-05). A pure-NVIDIA MTP pair will require either
NVIDIA shipping a Nemotron-Nano MTP variant in a future release,
or the project training its own EAGLE3 head against
`Llama-3.1-Nemotron-Nano-8B-v1` or `Llama-3.1-8B-Instruct-NVFP4`
(both were on disk in 2026-05).

### 5.7 Summary table -- what runs on 24 GB

For this project's RTX PRO 4000 Blackwell, the 2026-05 shortlist:

| Provider | Target | Drafter | Total VRAM est. | Status (2026-05) |
|---|---|---|---|---|
| Google | `gemma-4-E2B-it` (5 GB BF16) | `gemma-4-E2B-it-assistant` (0.15 GB) | ~6 GB + KV | downloaded |
| Google | `gemma-4-E4B-it` (15 GB BF16) | `gemma-4-E4B-it-assistant` (0.15 GB) | ~16 GB + KV | drafter downloaded; target not on disk |
| Google + NVIDIA | `nvidia/Gemma-4-26B-A4B-NVFP4` (17.5 GB) | `gemma-4-26B-A4B-it-assistant` (0.8 GB) | ~19 GB + KV | downloaded, **prime candidate** |
| Google + NVIDIA | `nvidia/Gemma-4-31B-IT-NVFP4` (~18 GB) | `gemma-4-31B-it-assistant` (0.9 GB) | ~20 GB + KV (tight) | drafter downloaded; target not on disk |
| Alibaba (community) | `Qwen3.6-27B-Text-NVFP4-MTP` (18.3 GB, built-in MTP) | -- | ~19 GB + KV | downloaded, **prime candidate** |
| NVIDIA (first-party) | `NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (67 GB, built-in MTP) | -- | ~70 GB | out of scope (datacenter only) |
| NVIDIA (first-party) | `meta-llama/Llama-3.3-70B-Instruct` (~40 GB NVFP4) | `nvidia/Llama-3.3-70B-Instruct-Eagle3` (3.2 B) | ~45 GB | out of scope (target too big) |
| DeepSeek | DeepSeek-V3 (~671 B) | built-in MTP | ~330 GB+ | out of scope |

The two **prime candidates** in 2026-05 were
`nvidia/Gemma-4-26B-A4B-NVFP4` + assistant (a lossless speedup, 2-3x
by the vendor's figures) and `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`
(self-contained, no drafter to manage). What was eventually measured
here is the Qwen3.8-27B built-in head (Sec. 7.1).

A live caveat for the Google + NVFP4 path: the target's BF16
distribution is not bit-exact equal to the NVFP4-quantised target's
distribution -- and the assistant was trained against BF16. The
verification step (Sec. 6) is *correct in expectation* (rejection
sampling matches the post-quantization target distribution by
construction), but **acceptance rate may drop** because the
drafter's proposals are calibrated to the BF16 target's logits, not
the quantised one's. This is the same issue that motivates training
EAGLE drafts on the same precision as the eventual serve. No
published numbers yet for the BF16-drafter + NVFP4-target pair;
probing will tell us.

---

## 6. Why MTP is "lossless" -- the verification guarantee

The phrase "identical quality" in Google's press release is not
marketing. It comes from a small but non-obvious fact about
rejection sampling, due to Leviathan et al. 2022:

> **If the target distribution is `q` and you sample `x` from a
> proposal `p`, then accept with probability `min(1, q(x)/p(x))` and
> on rejection sample once from `(q - p)+ / sum((q - p)+)`, the
> overall distribution of accepted samples is exactly `q`.**

In English: even though the drafter `p` proposes tokens, the
*distribution* of tokens that survive the accept/reject step is
exactly what `q` (the target) would have produced on its own. The
drafter cannot bias the output; it can only fail to predict useful
proposals and waste its forward passes.

Concretely, per drafted token `d_i`:

```
    q_i = target's probability of d_i given the prefix
    p_i = drafter's probability of d_i given the prefix
    accept with probability min(1, q_i / p_i)
    if rejected: resample one token from the
                 "residual" distribution
                   r(t) = max(0, q(t) - p(t))
                   r normalised to sum to 1
```

A few consequences worth internalising:

- **Greedy decode is a special case.** When the target samples
  greedily (temperature 0), acceptance reduces to "did the drafter
  predict the target's argmax?". Yes -> accept; no -> reject and
  emit the target's argmax. In exact arithmetic the output is
  identical to the unaccelerated path. On a GPU the verifier scores
  K+1 positions at once, and floating-point results can depend on such
  batch shapes, so near-ties can in principle resolve differently.
  What this project's data can show is limited: vLLM's log records
  output lengths, not text, and in 27 of the 34 temperature-0 pairs
  both arms stopped at the `max_tokens` limit, where equal lengths say
  nothing. In the 7 pairs where length is informative, 3 (AutoRound)
  and 6 (NVFP4) had a different length with MTP on than off. So at
  least 3 and 6 of 34 outputs changed; how many of the other 27 did is
  unknown. The MTP-on and MTP-off runs also differed in day, image and
  checkpoint copy, so the cause is not known either (Sec. 7.1).

- **Temperature sampling is also a special case.** With softmax
  temperatures applied to both target and drafter, the same
  rejection rule reproduces the target's sampled distribution
  exactly. No "drafter style bleed-through".

- **Better drafter -> more acceptance, not different output.** A
  bad drafter doesn't make the model dumber; it just makes the
  scheme slower. In the limit of a useless drafter (uniformly
  random), every token gets rejected, you get one fresh token per
  pass from the target, and you have paid an extra drafter forward
  pass for nothing.

- **Caveat: the target's *distribution* is preserved, not its
  RNG state.** If your application reads token positions out of a
  reproducible seed, that seed sequence won't match the
  unaccelerated path on a per-position basis. The marginal output
  distribution is identical; the specific samples differ. For
  agents this is invisible. At `temperature: 0` argmax is
  deterministic, so the output is identical in exact arithmetic; on a
  GPU it can still change (previous bullet), so do not expect bit-exact
  reproduction when MTP is switched on or off.

This is why a vendor can ship MTP behind a flag and not call it
"a different model". It is the same model, served faster.

---

## 7. MTP on this project

### 7.1 What MTP measured on this card

The Qwen3.8-27B builds served on this project (AutoRound W4A16 and
NVFP4 bodies, both prepared with int8 vocabulary tensors and a
40,960-token draft head; built-in MTP head, `num_speculative_tokens` 3)
were run over the bench's 40 short latency prompts once with MTP on and
once with MTP off. Because the prompts are the same in both arms, the
comparison is **paired**: one speedup per prompt.

| Build | Prompts (pairs) | Geometric mean of per-prompt speedups (95 % CI) | Median (95 % CI) | Range |
|---|---|---|---|---|
| AutoRound (W4A16) | 34 | **2.52x** (2.42-2.61) | 2.54x (2.40-2.68) | 2.01-3.16x |
| NVFP4 | 34 | **2.60x** (2.50-2.70) | 2.64x (2.38-2.80) | 2.10-3.20x |

What this buys and what it costs: decode about 2.5x faster on these
prompts, for 0.60 GiB more weights and a KV pool 24 % smaller (measured
on the NVFP4 build), which limits the context the build can serve
(Sec. 9.3).

How to read it:

- **Definition.** Per request, decode rate = generated tokens /
  (elapsed time - the run's median time to first token). Generated
  tokens and elapsed time come from vLLM's per-request log lines; the
  elapsed time runs from the request's arrival to its end, so it also
  contains any queueing and the prefill. The subtracted time to first
  token is not a per-request engine value but the client harness's median
  over the run's prompts 2-40: 70.7 ms (AutoRound) and 72.3 ms (NVFP4)
  with MTP on, 60.1 and 84.5 ms with MTP off. Speedup = MTP-on rate /
  MTP-off rate for the same prompt.
- **Pairing.** The engine log carries no prompt identifier.
  `perf_engine_runs.py` assigns log records to prompts in send order,
  using each prompt's `max_tokens` (a request that stopped on the length
  limit must have generated exactly that many tokens). The
  reconstruction is consistent: the prompt-token counts of the two arms
  match in all 34 pairs; no record was set aside in the MTP-on runs, and
  1 (NVFP4) and 8 (AutoRound) records of other clients were set aside in
  the MTP-off runs (the script lists them).
- **n and exclusions.** 40 prompts per arm; the first prompt is
  excluded (its engine time contains 11-15 s of first-request
  initialisation), as are prompts with fewer than 64 generated tokens
  in either arm, leaving 34 pairs. That filter depends on the outcome.
  With all 39 prompts the geometric means are 2.59x (2.48-2.70,
  AutoRound) and 2.68x (2.56-2.81, NVFP4), so the result does not hinge
  on it.
- **What the prompts represent.** The 40 prompts are a fixed,
  hand-written set of short English questions with at most 256 output
  tokens, run at temperature 0 with reasoning on. The intervals describe
  a notional population of similar prompts under these conditions
  ([statistics-primer.md](statistics-primer.md) Sec. 1); they say
  nothing about long outputs, other content or other temperatures.
- **Intervals.** Geometric-mean interval: percentile bootstrap over
  prompts (B = 10000, seed 20260927); median interval: order
  statistics (primer Sec. 8 and 9). They cover prompt-to-prompt
  variation only: each arm is **one run**, so run-to-run variation is
  not in them.
- **Conditions.** Temperature 0, reasoning on at `medium`, max_tokens
  16-256 per prompt. MTP on: 2026-09-21, prepared checkpoints in
  place, port 11435, image `localhost/devai-vllm` (vLLM 0.28.0; digest
  not recorded), ctx 131072 (AutoRound) and 98304 (NVFP4). MTP off:
  2026-09-22, the derived `-devai` copies on the vllm-devai backend
  (image `sha256:0a94e7982b85...`, vLLM 0.28.0), ctx 131072 and
  118784.
- **Confounders** (primer Sec. 4.3). Besides MTP, the arms differ in
  day, image tag, checkpoint copy, NVFP4 context and run order. How
  large is run-to-run variation? Single bench runs of eight models three
  days apart differed by -1.6 % to +3.8 % (implied single-run
  coefficient of variation about 1.3 %, 95 % CI roughly 0.8-2.6 %), and
  runs of one model differed by up to 13 % when image and context also
  changed (`llm-tokens-and-speed.md` Sec. 7; primer Sec. 9, "the unit of
  replication"). That is far less than a 2.5x effect, so the size of the
  speedup is robust; its second significant digit is not.
- **What varies.** Without MTP the decode rate hardly depends on the
  prompt (median 38.9 tok/s AutoRound, 36.4 NVFP4 over the 34 pairs;
  SD 0.35-0.38 tok/s across prompts). With MTP it does (SD about
  12 tok/s), because the share of drafted tokens accepted differs from
  prompt to prompt. vLLM counted 66 % of drafted tokens accepted over
  these runs, about 3.0 tokens per verification step out of a possible
  4 (Sec. 5.2).

Other MTP numbers quoted in this repo, and what they are:

- **Two different "95 tok/s" figures.** (a) 95.2 tok/s:
  Qwen3.8-27B-MTP-devai-NVFP4, `::nothink::mtp`, 2026-09-22, the engine
  median of 6 replicate requests of **one** code prompt (range
  85.3-97.5; ratio of summed tokens to summed time 93.8). One prompt,
  one run. (b) 95.13 tok/s: the bench harness's median for the
  **AutoRound** build with MTP on over the 40 short prompts (2026-09-21,
  characters/4 token count; the engine median of the same run is
  101.7 tok/s). CLAUDE.md's "95 tok/s single" for the AutoRound build is
  (b); the "37 vs 95 tok/s" quoted for the NVFP4 build is (a).
- **"37 vs 95 tok/s"**: 36.8 is the bench's MTP-off median on the short
  prompts with reasoning on (characters/4 token count), 95 the code
  prompt (a). Different prompts, reasoning modes and token counters:
  an unpaired, confounded comparison that happens to land near the
  paired result. Quote the table above instead.
- **Other workloads.** On a code prompt with thinking unintentionally
  on (router policy bug, 2026-09-22; 3 runs of 5-6 replicates) MTP-on
  medians were 73.0-78.4 tok/s, about 2.0-2.2x the MTP-off rate *if*
  that rate is as prompt-independent there as above (an assumption; no
  MTP-off run of that prompt exists). On the aiagent sentiment workload
  (2026-09-26, NVFP4 build, temperature 0.7, prompts of about 330
  tokens) vLLM counted 46 % of drafted tokens accepted. That run differs
  from the bench prompts in task, temperature, prompt length and build,
  so the lower rate cannot be attributed to content alone; and
  acceptance is not speedup: with no MTP-off control, the speedup on
  that workload was not measured.
- The picker launches every MTP-capable row with `::mtp`, but the bench
  never sends `::mtp`: the picker's TPS column is the MTP-off number.

Reproduce with `scripts/stats/perf_engine_runs.py` (reads
`/var/cache/devai/logs/devai-vllm.log` and `devai-vllm-devai.log`; the
paired table, the all-39 sensitivity, per-position and pooled acceptance,
the output-length comparison of Sec. 6 and the 2.0-2.2x ratios are in its
output), `perf_phases.py` (the memory cost in Sec. 9.3) and `perf_ts.py`
(the 46 % sentiment-workload acceptance).

### 7.2 How the router serves MTP

MTP has been supported end to end since 2026-05 (the design notes are
kept in Appendix A):

- **Catalog.** A model family's entry in `scripts/model-families.yaml`
  (an `hf_repos:` entry or a derived row) may carry an `mtp:` block:
  `method` (`mtp`, `qwen3_5_mtp`, ...), `num_speculative_tokens`, and a
  `drafter` repo for an external drafter. `generate-catalog.py` copies it
  into `deploy/models.yaml`. The Qwen3.8 derived rows declare
  `method: mtp`, `num_speculative_tokens: 3`.
- **Probe.** For a model with an `mtp:` block the HF prober
  (`scripts/_probe_hf_common.py`) also launches with MTP and records per
  cell whether it still fits; a cell that does not records
  `mtp_fits=false`.
- **Router.** `peelControlSuffixes` strips `@<ctx>`, `::mtp` /
  `::nomtp` and `::<reasoning>` from the model name in **any order** (it
  peels whichever suffix is trailing until none is left).
  `parseMTPOverride` reads the MTP token. For vLLM the router emits
  `--speculative-config '<json>'` (`vllmSpeculativeJSON`), for SGLang the
  `--speculative-*` flags (`sglangSpeculativeArgs`). The MTP setting is
  tracked per backend (`currentSpec`, compared with `specEqual`), so
  changing it recreates the container, like a model or `@<ctx>` change.
  Reasoning on + MTP on an inline-reasoning model is refused with HTTP
  400 ([vllm #34650](https://github.com/vllm-project/vllm/issues/34650)).
- **Picker.** `scripts/model-picker.py` (`_has_mtp`) launches every row
  that supports MTP with `::mtp`; the ON/OFF sub-modal was removed on
  2026-09-22. An inline-reasoning row that supports MTP is launched
  `::nothink::mtp`.

Example: `Qwen3.8-27B-MTP-devai-NVFP4::nothink::mtp@118784` -> ctx
118784, reasoning off, MTP on; the router recreates vllm-devai with
`--speculative-config` if the running container was launched without
it.

### 7.3 What does NOT need to change

For completeness, two things stay the same:

- **The OpenAI / Anthropic wire format.** MTP is invisible to the
  client. `messages`, `tool_calls`, `reasoning_content`, streaming
  -- all unchanged. See
  [`openai-api-and-streaming.md`](openai-api-and-streaming.md).
- **The reasoning / tool parsers.** A model that streams `<think>`
  tags emits the same kinds of tags whether MTP is on or off (the
  verifier is the same model), so the parsers need no change. Same
  goes for tool calls and structured output. There is **one known issue** in vLLM
  combining MTP with structured output + reasoning mode -- see
  [vllm #34650](https://github.com/vllm-project/vllm/issues/34650);
  the `</think>` token detection drops under MTP. Tracked, not
  yet patched -- avoid MTP for any agent path that relies on
  reasoning-content separation until the upstream fix lands.

---

## 8. Picking `num_speculative_tokens`

The recommendation from vLLM's Gemma 4 recipe, NVIDIA-measured on
A100/H100:

| Model | num_speculative_tokens | Note |
|---|---|---|
| E2B | 2 | small drafter, low draft cost; K=2 is the floor |
| E4B | 4 | drafter is more accurate; K can grow |
| 26B-A4B | 4 | MoE target verifies fast |
| 31B | 4-8 | dense target verifies more slowly per pass; higher K amortises better |

Higher K = more drafter work per round, more potential acceptance,
but also more *wasted* drafter forward passes when the verifier
rejects early. The optimum is workload-dependent: more K on
repetitive code, less on novel prose. Google's heuristic
adaptation in HuggingFace Transformers
(`num_assistant_tokens_schedule="heuristic"`) raises K by 2 on full
acceptance and lowers by 1 on rejection; vLLM does not (yet) ship
the adaptive scheduler -- you pick K statically per model.

For first probing on this project's RTX PRO 4000 Blackwell (2026-05
plan):
- `Gemma-4-26B-A4B-NVFP4` with assistant: start at `K=4`.
- `Qwen3.6-27B-Text-NVFP4-MTP`: start at `K=3` (the value the model
  card reports as best; external).
- Re-probe at K=2 and K=6 if first results are interesting.

The Qwen3.8-27B builds measured in Sec. 7.1 ran at K=3; no other K was
measured here.

---

## 9. Limits and failure modes

### 9.1 Acceptance rate collapses under high-novelty content

The drafter is trained on a particular distribution. On data far
from that distribution (a non-Latin language the drafter has
barely seen, a custom DSL, exotic JSON schemas), acceptance drops
toward zero. You still get correctness (the verifier always
catches it), but the speedup vanishes and you have paid drafter
forward-pass cost for nothing. Adaptive K schedules help; a
"detect and disable" fallback (drop K to 0 after N consecutive
rejections) is on the SGLang roadmap and not yet shipped.

Measured here (one run each): vLLM counted 66 % of drafted tokens
accepted on the bench's short English prompts and 46 % on the aiagent
sentiment workload (Sec. 7.1). This does not measure what content alone
does: the two runs also differ in temperature (0 vs 0.7), prompt length
(tens vs about 330 tokens), task and build. And acceptance is not
speedup; the sentiment workload's speedup was not measured.

### 9.2 Quantization mismatch can degrade acceptance

A drafter trained against the BF16 target sees slightly different
logits when paired with an NVFP4 target. Output remains
correct-by-construction (Sec. 6), but acceptance can drop several
percentage points (expected from how drafters are trained; not
measured here). The right answer is to (re)train the drafter
on the *quantised* target's logits -- which is exactly what
production EAGLE3 recipes prescribe. Until then, expect Gemma 4
+ NVFP4 to show somewhat lower acceptance than Gemma 4 + BF16.

### 9.3 MTP costs memory: more weights, a smaller KV pool

The drafter does fewer FLOPs, but it needs memory. For Gemma 4 the
drafter *shares* the target's KV blocks, which is a large saving. A
built-in head is not free either. Measured for the prepared
Qwen3.8-27B NVFP4 build on vllm-devai (vLLM log, 2026-09-22; same
image, `--gpu-memory-utilization` 0.96 and `--max-model-len` 118784 in
both arms, graphs compiled at launch; 7 launches with MTP on and 3
off, each arm giving the same values every time):

| | Weights loaded | KV memory available | KV pool |
|---|---|---|---|
| MTP off | 15.53 GiB | 4.99 GiB | 156,335 tokens |
| MTP on | 16.13 GiB | 4.28 GiB | 118,784 tokens |

MTP costs 0.60 GiB of weights and about 13 % more KV bytes per cached
token (38.7 vs 34.3 KB: KV memory divided by pool tokens), which fits
the MTP layer keeping KV of its own; the log does not break the
difference down. Together they shrink the KV pool by 24 %. The pool is
what bounds the context: this build's 118,784-token context is the
largest that fits with MTP on (found with an exact-context probe),
while the same settings without MTP leave a pool of 156,335 tokens.
With several sequences in flight the same pool also bounds
`--max-num-seqs`. The verifier's K extra positions per round are a
further, much smaller cost (derived: 3 positions per sequence).
Reproduce with `scripts/stats/perf_phases.py`
(`mtp_memory_cost_nvfp4_devai_2026_09_22`).

### 9.4 Streaming with `reasoning_parser` is fragile

See vllm #34650 above. The `</think>` close token detection
mis-fires under MTP because the verifier sees a multi-token batch
where the reasoning parser expects single-token streaming.

### 9.5 Tool calling under MTP has not been widely probed

The literature is mostly about plain text. Tool-call structured
output relies on the parser detecting `<tool_call>` tags
character-by-character in the streamed output; the parser logic
inside vLLM/SGLang has been updated for MTP but the test surface
is small. **Probe before claiming this works on this project's
hardware.**

### 9.6 Cold-start adds drafter load time

For Gemma 4 the drafter is ~150 MB - 900 MB BF16; at the few GB/s
the weight-load phase achieves on this card (1.5-4.9 s for 14.5-16.1
GiB over all 37 vllm-devai launches,
[`nvfp4-coldstart.md`](nvfp4-coldstart.md) Sec. 1) that is well under
a second of extra load (derived, not measured). For built-in heads
the drafter weights are already in the target's checkpoint. Measured
for the Qwen3.8-27B NVFP4 build on vllm-devai (vLLM log, 2026-09-22):
model loading took 7.9-8.8 s with MTP on (7 launches) against
about 7.5 s off (3 launches), and MTP added one extra torch.compile
range of 3.2-3.4 s (`perf_phases.py`). Both are small next to the
whole cold start: router launch-to-ready medians range from 40-45 s
(8B-class NVFP4, 2026-04/05) to 125-271 s (these 27B builds, depending
on which caches were warm), see `nvfp4-coldstart.md` Sec. 1.

---

## 10. Practical recipe

Using MTP on this project needs no configuration beyond the catalog:

```
    # 1. A row whose catalog entry has an `mtp:` block (and whose probe
    #    cell did not record mtp_fits=false) is always launched by the
    #    picker as <name>::mtp@<ctx>; the router recreates the backend
    #    with --speculative-config.
    devai-agent            # pick the row; MTP is on

    # 2. Or address it directly:
    #    model = "Qwen3.8-27B-MTP-devai-NVFP4::nothink::mtp@118784"
    #    (inline-reasoning rows need ::nothink with ::mtp, see Sec. 7.2)
```

To **measure** the speedup for a new model or workload, use a paired
design rather than comparing two unrelated numbers
([statistics-primer.md](statistics-primer.md) Sec. 9):

- the same prompt list in both arms (`<name>@<ctx>` vs
  `<name>::mtp@<ctx>`), temperature 0 for the rate itself and, if the
  workload samples, a second pass at its own temperature (acceptance
  depends on sampling and content);
- engine-counted tokens (request `stream_options.include_usage`, or
  read vLLM's per-request `Request finished` log lines); discard the
  first request after each launch;
- fixed `max_tokens` large enough that most requests stop on length;
- several launches per arm, alternated: A B B A gives two per arm,
  A B B A A B B A four, so drift and launch-to-launch variation are not
  confounded with MTP;
- report the geometric mean of per-prompt speedups with a bootstrap
  interval, and say what it covers.

`scripts/stats/perf_engine_runs.py` performs this analysis on the
2026-09-21/22 logs (Sec. 7.1). For an external drafter that the catalog
does not describe, the `RecoveryFlags` JSON in Appendix A.1 still works
as an operator override.

---

## 11. Summary

- Decode is bandwidth-bound; one forward pass produces one token,
  and the GPU spends most of its time reading weights.
- **MTP / speculative decoding** breaks the 1-token-per-pass
  barrier by drafting K tokens with a small fast drafter and
  verifying them in one big-model forward pass. Acceptance is
  guaranteed-correct by rejection sampling.
- Two architectures in 2026: **external drafter** (Gemma 4, EAGLE,
  Medusa) and **built-in MTP head** (DeepSeek V3, Qwen3.6, Qwen3.8).
  The verification math is identical; the difference is where the
  drafter's weights live.
- Vendors and papers report about 1.7-3x (external). On this
  project's RTX PRO 4000 Blackwell, the Qwen3.8-27B builds with their
  built-in head (K=3) decoded **2.52x** (AutoRound, 95 % CI 2.42-2.61)
  and **2.60x** (NVFP4, 2.50-2.70) faster than without MTP: paired
  over 34 prompts of the bench's fixed set of short prompts at
  temperature 0, one run per arm, per-prompt range 2.0-3.2x; 2.59x and
  2.68x with all 39 prompts (Sec. 7.1). vLLM accepted 66 % of drafted
  tokens on those prompts; on the aiagent sentiment workload (other
  temperature, prompt length and build) it accepted 46 %, and the
  speedup there was not measured.
- MTP costs memory: on the prepared Qwen3.8-27B NVFP4 build, 0.60 GiB
  more weights and a 24 % smaller KV pool, which limits the servable
  context (Sec. 9.3).
- The router supports MTP since 2026-05: an `::mtp` suffix, a catalog
  `mtp:` block, `--speculative-config` emission and a recreate when the
  MTP setting changes (Sec. 7.2). The picker launches every MTP-capable
  row with `::mtp`; the bench's TPS column is still measured without
  MTP.
- "Lossless" holds for the output *distribution* (Sec. 6). Specific
  samples differ when temperature > 0 due to RNG ordering, and at
  temperature 0 outputs can still change on a GPU: here at least 3
  (AutoRound) and 6 (NVFP4) of 34 outputs had a different length with
  MTP on, out of the 7 pairs in which length could show it; the rest
  could not be checked, and the cause is not isolated.

---

## 12. References

### Foundational papers

- Leviathan, Y., Kalman, M., & Matias, Y. (2022). *Fast Inference
  from Transformers via Speculative Decoding.*
  [arXiv:2211.17192](https://arxiv.org/abs/2211.17192). The
  original rejection-sampling-based speculative decoding scheme;
  cited by every subsequent paper in this list.
- Cai, T. *et al.* (2024). *Medusa: Simple LLM Inference
  Acceleration Framework with Multiple Decoding Heads.*
  [arXiv:2401.10774](https://arxiv.org/abs/2401.10774).
- Li, Y. *et al.* (2024). *EAGLE: Speculative Sampling Requires
  Rethinking Feature Uncertainty.*
  [arXiv:2401.15077](https://arxiv.org/abs/2401.15077). EAGLE3
  is the 2025 successor; same group.
- DeepSeek AI (2024). *DeepSeek-V3 Technical Report.*
  [arXiv:2412.19437](https://arxiv.org/abs/2412.19437). Sec. 2.1.2
  is the MTP architecture description.

### Provider docs

- Google (2026-05). *Accelerating Gemma 4: faster inference with
  multi-token prediction drafters.*
  <https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/>.
  Release announcement.
- Google AI for Developers. *Gemma 4 Multi-Token Prediction (MTP)
  using HuggingFace Transformers.*
  <https://ai.google.dev/gemma/docs/mtp/mtp>. Practical usage page.
- vLLM Recipes. *Gemma 4 Usage Guide.*
  <https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html>.
  The `--speculative-config` JSON examples and per-model
  recommendations.
- vLLM. *MTP (Multi-Token Prediction).*
  <https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/>.
- vLLM. *Speculative Decoding (umbrella docs).*
  <https://docs.vllm.ai/en/latest/features/speculative_decoding/>.
- SGLang. *Speculative Decoding.*
  <https://docs.sglang.io/advanced_features/speculative_decoding.html>.
  Full `--speculative-*` flag inventory.

### Model checkpoints

- `nvidia/Gemma-4-26B-A4B-NVFP4`:
  <https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4>.
- `nvidia/Gemma-4-31B-IT-NVFP4`:
  <https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4>.
- `google/gemma-4-26B-A4B-it-assistant`:
  <https://huggingface.co/google/gemma-4-26B-A4B-it-assistant>.
- `google/gemma-4-31B-it-assistant`:
  <https://huggingface.co/google/gemma-4-31B-it-assistant>.
- `google/gemma-4-E2B-it-assistant`:
  <https://huggingface.co/google/gemma-4-E2B-it-assistant>.
- `google/gemma-4-E4B-it-assistant`:
  <https://huggingface.co/google/gemma-4-E4B-it-assistant>.
- `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`:
  <https://huggingface.co/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP>.
- `lujangusface/tw-eagle3-gemma4` (community EAGLE3 for Gemma-4-31B):
  <https://huggingface.co/blog/lujangusface/tw-eagle3-gemma4>.
- `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (first-party MTP):
  <https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4>.
- `nvidia/Llama-3.3-70B-Instruct-Eagle3` (NVIDIA-trained EAGLE3 head):
  <https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-Eagle3>.
- NVIDIA Nemotron 3 Super technical report:
  <https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf>.
- *Accelerating PayPal's Commerce Agent with Speculative Decoding:
  An Empirical Study on EAGLE3 with Fine-Tuned Nemotron Models* (2026):
  [arXiv:2604.19767](https://arxiv.org/abs/2604.19767). EAGLE3
  trained against `Llama-3.1-Nemotron-Nano-8B-v1`.

### Known issues

- [vllm #34650](https://github.com/vllm-project/vllm/issues/34650)
  -- structured output + reasoning + MTP triggers a `</think>`
  detection failure. Track before turning MTP on for any agent
  flow that consumes `reasoning_content`.

### Project-internal cross-references

- [`llm-tokens-and-speed.md`](llm-tokens-and-speed.md) Sec. 5-7 --
  prefill vs decode, the bandwidth-bound decode ceiling that MTP
  attacks.
- [`paged-attention-and-vllm-internals.md`](paged-attention-and-vllm-internals.md)
  Sec. 4 -- continuous batching, which is the kernel-level
  ingredient that lets the verifier batch its K-position
  proposals through one forward pass.
- [`attention-and-the-transformer.md`](attention-and-the-transformer.md)
  Sec. 6 -- KV cache mechanics; MTP-shared KV reuses these same
  blocks across two models in the same address space.
- [`nvfp4-coldstart.md`](nvfp4-coldstart.md) Sec. 2 -- the VRAM
  stack the drafter loads into.
- [`sampling-strategies.md`](sampling-strategies.md) Sec. 1-3 --
  the target-distribution `q` whose preservation Sec. 6 above
  proves.
- [`router.md`](router.md) -- the request rewrite chain (override
  parsing -> reasoning policy -> tool_choice promotion -> tool
  stripping -> ctx injection), including the `::mtp` / `::nomtp`
  override parser.
- [`backends.md`](backends.md) -- backend lifecycle, where any
  `--speculative-config` flag must be injected before container
  start (a backend recreate is required to change the MTP
  configuration, same as for `currentModel` and `currentContext`).
- [`bench-results.md`](bench-results.md) -- the bench's MTP-off tok/s
  numbers (dated single runs, characters/4 token counts).
- [statistics-primer.md](statistics-primer.md) Sec. 4.3 (confounding),
  Sec. 8 (quantiles and the median interval), Sec. 9 (per-request
  rates, what counts as a token, ratios and the bootstrap, drift and
  replication), Sec. 14 (reporting layout).
- `scripts/stats/perf_engine_runs.py` -- reproduces Sec. 7.1 from the
  persisted vLLM logs; `scripts/stats/perf_phases.py` -- the launch
  phases and the memory cost in Sec. 9.3; `scripts/stats/perf_ts.py` --
  the sentiment-workload acceptance.

---

## Appendix A. 2026-05 design notes

These are the design notes written before MTP support shipped in
2026-05, kept for history. Where they describe something as missing or
planned, Sec. 7.2 describes what exists. The `RecoveryFlags` route in
A.1 still works as an operator override for a drafter the catalog does
not describe.

### A.1 Minimum-viable: ride the `RecoveryFlags` escape hatch

`gpu-arbiter/main.go` already has a per-model CLI-args bag called
`RecoveryFlags`, sourced from `deploy/recovery-flags.json` and
appended verbatim to the backend container's entrypoint at launch.
Today that bag carries things like `--enforce-eager` for models
whose CUDA-graph workspace pushes them past 24 GB.

Speculative-decoding flags drop straight in. For
`Gemma-4-26B-A4B-NVFP4`, the recovery JSON entry would carry:

```
    "engine_flags": [
      "--speculative-config",
      "{\"method\":\"mtp\",\"model\":\"/models/gemma-4-26B-A4B-it-assistant\",\"num_speculative_tokens\":4}"
    ]
```

For `Qwen3.6-27B-Text-NVFP4-MTP` (built-in MTP head):

```
    "engine_flags": [
      "--speculative-config",
      "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":3}"
    ]
```

Pros:
- Zero Go code changes.
- Zero probe-cache schema changes.
- Per-model opt-in / opt-out via a single JSON file.

Cons:
- The drafter must be mounted into the vLLM container under
  `/models/...`. The compose file's volume mount already covers
  `VLLM_MODELS_DIR`, so this works as long as the drafter directory
  sits next to the target inside that tree -- which is where we
  just downloaded them.
- The probe's VRAM-fit data was measured *without* the drafter
  loaded. The drafter adds 150 MB - 900 MB of weight VRAM and some
  drafter-KV (small thanks to KV sharing). At 24 GB this is usually
  a no-op for fit but it is unmeasured -- you would learn whether
  it OOMs at long context only when you tried.
- The picker shows one row per `(model, backend)`. There's no way
  to expose an "MTP on / off" toggle in the UI without a code
  change.

This is the recommended path for an initial trial.

### A.2 Catalog + probe + entrypoint -- the clean implementation

A first-class MTP integration touches four places:

1. **`scripts/model-families.yaml`**. Add a new optional sub-block
   per `hf_repos:` entry recording the matched drafter and the
   recommended `num_speculative_tokens`. Example:

   ```yaml
       hf_repos:
         - repo: nvidia/Gemma-4-26B-A4B-NVFP4
           mtp:
             method: mtp
             drafter: google/gemma-4-26B-A4B-it-assistant
             num_speculative_tokens: 4
         - repo: sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP
           mtp:
             method: qwen3_5_mtp
             num_speculative_tokens: 3
   ```

   `generate-catalog.py` propagates this into `deploy/models.yaml`.

2. **`scripts/probe-vllm-reasoning.py`** (and SGLang counterpart).
   When a model declares an MTP block, the probe launches vLLM
   with `--speculative-config` so peak VRAM and `fits=true` reflect
   the drafter's footprint. Bump cache schema v2 -> v3; add a new
   per-cell field `mtp_overhead_gb` so downstream consumers can
   show "fits at 128K with MTP" alongside "fits at 256K without
   MTP". Probe-cache writers and readers must stay in lock-step --
   see [`probe-cache-schema-reviewer`](../scripts/probe-vllm-reasoning.py)
   for the project's standing guidance on schema drift.

3. **`gpu-arbiter/main.go`** entrypoints. Three small additions:

   - A `Speculative *configSpeculative` field on `configModel` and
     `launchConfig`, parsed from the catalog and the per-request
     suffix override (A.3 below).
   - In `vllmEntrypoint`, if `lc.Speculative != nil`, emit
     `--speculative-config '<json>'` after the parser flags and
     before `RecoveryFlags...`.
   - In `sglangEntrypoint`, the equivalent: emit
     `--speculative-algorithm`, `--speculative-num-steps`,
     `--speculative-num-draft-tokens`, `--speculative-eagle-topk`,
     and optionally `--speculative-draft-model-path` for external
     drafters.

   `containerRecreate` already tracks `currentModel` and
   `currentContext` -- add `currentSpec` so a request that toggles
   `::mtp` recreates the backend (each change is a recreate trigger
   the same way `@<ctx>` overrides already are).

4. **`scripts/model-picker.py`**. Add an MTP toggle in the
   post-select modal, mirroring the existing reasoning ON/OFF
   sub-modal. The picker emits `::mtp` or `::nomtp` as a suffix
   on the model name (analog to `::nothink`). (Superseded 2026-09-22:
   the toggle was removed again; a supporting row always emits
   `::mtp`, and an inline-reasoning MTP row emits `::nothink::mtp`.)

### A.3 Per-request override -- the `::mtp` suffix

The router already parses two suffixes on the model name:
`@<ctx>` (context cap) and `::<reasoning>` (e.g. `::nothink`). A
third suffix, `::mtp` / `::nomtp`, fits the same chain.

Parsing order needs to be stable. The order at the time was:
`parseCtxOverride` first (strips `@<ctx>`), `parseReasoningOverride`
second (strips `::<reasoning>`). MTP slots in between -- it is more
specific than the reasoning override and shares the `::` separator.
The natural rule: any `::<token>` that matches `mtp` or `nomtp` is
the MTP override; anything else falls through to the reasoning
parser. (Superseded: the shipped `peelControlSuffixes` strips the three
suffixes in any order, because some clients append `::<reasoning>`
after `@<ctx>`; see Sec. 7.2 and `router.md`.)

The picker emits, e.g.:

```
    gemma-4-26B-A4B-NVFP4::mtp@131072
```

Router parses: `@131072` -> ctx=128K; `::mtp` -> MTP on for this
session. `containerRecreate` sees `currentSpec` differs, recreates
the vLLM container with the MTP flag, and serves.

---

---

## Changes to this page (2026-09-27)

Earlier versions of this page stated, and this version corrects:

- a general "2-3x decode speedup" (external figures are 1.7-3x; what
  was measured here is 2.52x and 2.60x on one fixed set of short
  prompts, Sec. 7.1), and bit-exact identical output at temperature 0
  (not guaranteed on a GPU; at least 3 and 6 of 34 outputs changed
  here, Sec. 6);
- an MTP memory cost that was "small (~1 GB)" or "sub-1 % of pool
  capacity" (the engine log shows +0.60 GiB of weights and a KV pool
  24 % smaller, Sec. 9.3);
- "3-4 mean accepted tokens per draft pass" from per-position rates
  whose definition was not stated (Sec. 5.2), and the config key
  `mtp_num_layers` (it is `mtp_num_hidden_layers`);
- a drafter load time compared against a 30-60 s cold start (the 27B
  builds here take 125-271 s; Sec. 9.6).

The 2026-05 design notes that used to open Sec. 7 are now Appendix A;
Sec. 7.2 describes what shipped, including the order-independent
suffix parsing.
