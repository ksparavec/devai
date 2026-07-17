# Multi-token prediction (MTP) -- a 2-3x decode speedup with the same model

This page covers the trick that lets a model emit several tokens per
forward pass instead of one, without changing the model's outputs at
all. The canonical name on Google's product pages is **multi-token
prediction (MTP)**; the canonical name in the literature is
**speculative decoding**. They are *almost* the same thing -- they
differ in *who proposes the next K tokens* (a separate drafter model
vs. extra heads stitched into the main model), but the verification
math is identical.

Both names show up in vLLM / SGLang / HuggingFace / Ollama flag docs
because both shapes are live in 2026. This doc explains which one
each provider ships and what it would take to wire MTP into this
project's router.

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
ceiling is `640 GB/s / 5.1 GB ~ 125 tok/s` and the measured number
is `98.3 tok/s` -- ~78 % bandwidth utilisation. There is no way to
push past that ceiling *as long as the model emits one token per
forward pass*. The GPU is not compute-starved; it is memory-bandwidth
starved.

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
bandwidth-bound; you pay the same ~5 GB-per-step weight read whether
you process 1 or 16 positions in that step (small K does not push
into compute-bound territory). The drafter's K serial passes are
cheap because the drafter is tiny.

End-to-end: ~2-3x decode speedup at typical acceptance rates.

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
`mtp_num_hidden_layers=1` arrangement on a 27 B target.

The two shapes are summarised:

| Aspect | External drafter | Built-in MTP head |
|---|---|---|
| Where weights live | separate repo (`-assistant`, `-eagle3`) | same checkpoint as target |
| Extra VRAM cost | drafter weights + drafter KV (sharable) | small (~1 GB), already counted in checkpoint |
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

Google's published headline: **up to 3x decoding speedup on NVIDIA
RTX PRO 6000**, with bit-exact identity to the unaccelerated path
(see Sec. 6). On Apple Silicon at batch 4-8 they report ~2.2x for
the 26 B MoE. Same model, same outputs, ~half the wait.

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

The official acceptance rate on MTP-1 is **>80 %**, yielding
**~1.8x** decode throughput. vLLM flag:

```
    --speculative-config '{"method": "deepseek_mtp",
                           "num_speculative_tokens": 1}'
```

V3 is 671 B total / 37 B active -- not even close to fitting on a
24 GB card. Listed here for completeness; the project's catalog
does not include DeepSeek-V3 base.

### 5.2 Qwen3.6 -- built-in MTP at 24 GB-friendly sizes

Qwen3.6 (released 2026-Q1) ships **`mtp_num_layers=1`** -- a single
MTP module that can be applied recursively up to N speculative
steps. Per-position acceptance is reported as ~87 % / 72 % / 61 %
for positions 1/2/3 with `num_speculative_tokens=3`, giving 3-4
mean accepted tokens per draft pass.

The catch: stock NVFP4 quantization scripts **drop the MTP head**
because `AutoModelForCausalLM.from_pretrained` doesn't load it.
Community quants restore it in BF16. The relevant 24 GB-class
checkpoint:

| Repo | Size | KV-pool headroom @256K |
|---|---|---|
| `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` | 18.3 GB (NVFP4 weights + ~850 MB BF16 MTP head) | fits with `--kv-cache-dtype fp8 --max-num-seqs 2` |

The author's reported speedup: **1.74x** on long-form decode
(207 vs 119 tok/s) with `num_speculative_tokens=3`. vLLM flag:

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
and reported a **1.72x speedup** -- slightly slower than Google's
official assistant, but trained with a fraction of the compute.

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
accepted tokens, beating DeepSeek-R1 in NVIDIA's own benchmarks)
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
report 22-49 % throughput gain at gamma=3 -- but that draft head
is not publicly released as of 2026-05.

**Net for this project:** the NVIDIA-branded fit-the-card MTP
path today goes through *NVIDIA's NVFP4 quantization of Google's
Gemma 4* (`nvidia/Gemma-4-26B-A4B-NVFP4`, already in the catalog
and downloaded). A pure-NVIDIA MTP pair will require either
NVIDIA shipping a Nemotron-Nano MTP variant in a future release,
or the project training its own EAGLE3 head against
`Llama-3.1-Nemotron-Nano-8B-v1` or `Llama-3.1-8B-Instruct-NVFP4`
(both already on disk per `ls /var/cache/devai/vllm/`).

### 5.7 Summary table -- what runs on 24 GB

For this project's RTX PRO 4000 Blackwell, the 2026-05 shortlist:

| Provider | Target | Drafter | Total VRAM est. | Status today |
|---|---|---|---|---|
| Google | `gemma-4-E2B-it` (5 GB BF16) | `gemma-4-E2B-it-assistant` (0.15 GB) | ~6 GB + KV | downloaded |
| Google | `gemma-4-E4B-it` (15 GB BF16) | `gemma-4-E4B-it-assistant` (0.15 GB) | ~16 GB + KV | drafter downloaded; target not on disk |
| Google + NVIDIA | `nvidia/Gemma-4-26B-A4B-NVFP4` (17.5 GB) | `gemma-4-26B-A4B-it-assistant` (0.8 GB) | ~19 GB + KV | downloaded, **prime candidate** |
| Google + NVIDIA | `nvidia/Gemma-4-31B-IT-NVFP4` (~18 GB) | `gemma-4-31B-it-assistant` (0.9 GB) | ~20 GB + KV (tight) | drafter downloaded; target not on disk |
| Alibaba (community) | `Qwen3.6-27B-Text-NVFP4-MTP` (18.3 GB, built-in MTP) | -- | ~19 GB + KV | downloaded, **prime candidate** |
| NVIDIA (first-party) | `NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (67 GB, built-in MTP) | -- | ~70 GB | out of scope (datacenter only) |
| NVIDIA (first-party) | `meta-llama/Llama-3.3-70B-Instruct` (~40 GB NVFP4) | `nvidia/Llama-3.3-70B-Instruct-Eagle3` (3.2 B) | ~45 GB | out of scope (target too big) |
| DeepSeek | DeepSeek-V3 (~671 B) | built-in MTP | ~330 GB+ | out of scope |

The two **prime candidates** for this project are
`nvidia/Gemma-4-26B-A4B-NVFP4` + assistant (lossless 2-3x speedup
on a quality-tier model) and `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`
(self-contained, no drafter to manage).

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
  emit the target's argmax. Output is bit-exact identical to the
  unaccelerated path.

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
  agents this is invisible; for `temperature: 0` evaluation this is
  often bit-exact (because argmax is deterministic).

This is why a vendor can ship MTP behind a flag and not call it
"a different model". It is the same model, served faster.

---

## 7. How this project's router supports MTP

**Update 2026-05 (catalog-crystalline-beaver):** the clean
implementation outlined below in Sec. 7.2 has shipped. The picker
opt-in (`::mtp` suffix), the catalog `mtp:` block, the
`--speculative-config` emission, the `currentSpec` recreate
trigger, and the reasoning+MTP+inline guard for vllm#34650 are all
in place. See `gpu-arbiter/main.go` (`parseMTPOverride`,
`vllmSpeculativeJSON`, `sglangSpeculativeArgs`, `specEqual`,
`specLabel`), `scripts/model-picker.py` (`_has_mtp`, MTP sub-modal),
and `scripts/_probe_hf_common.py` (the per-cell MTP overhead probe).
The `RecoveryFlags` escape hatch in Sec. 7.1 remains available for
operator-level overrides but is no longer the recommended path.

### 7.1 Minimum-viable: ride the `RecoveryFlags` escape hatch

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

### 7.2 Catalog + probe + entrypoint -- the clean implementation

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
     suffix override (Sec. 7.3 below).
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
   on the model name (analog to `::nothink`).

### 7.3 Per-request override -- the `::mtp` suffix

The router already parses two suffixes on the model name:
`@<ctx>` (context cap) and `::<reasoning>` (e.g. `::nothink`). A
third suffix, `::mtp` / `::nomtp`, fits the same chain.

Parsing order needs to be stable. The current order is:
`parseCtxOverride` first (strips `@<ctx>`), `parseReasoningOverride`
second (strips `::<reasoning>`). MTP slots in between -- it is more
specific than the reasoning override and shares the `::` separator.
The natural rule: any `::<token>` that matches `mtp` or `nomtp` is
the MTP override; anything else falls through to the reasoning
parser.

The picker emits, e.g.:

```
    gemma-4-26B-A4B-NVFP4::mtp@131072
```

Router parses: `@131072` -> ctx=128K; `::mtp` -> MTP on for this
session. `containerRecreate` sees `currentSpec` differs, recreates
the vLLM container with the MTP flag, and serves.

### 7.4 What does NOT need to change

For completeness, two things stay the same:

- **The OpenAI / Anthropic wire format.** MTP is invisible to the
  client. `messages`, `tool_calls`, `reasoning_content`, streaming
  -- all unchanged. See
  [`openai-api-and-streaming.md`](openai-api-and-streaming.md).
- **The reasoning / tool parsers.** A model that streams `<think>`
  tags streams them at the same positions whether MTP is on or
  off (the verifier is the same model). Same goes for tool calls
  and structured output. There is **one known issue** in vLLM
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

For first probing on this project's RTX PRO 4000 Blackwell:
- `Gemma-4-26B-A4B-NVFP4` with assistant: start at `K=4`.
- `Qwen3.6-27B-Text-NVFP4-MTP`: start at `K=3` (the model card's
  measured-best value).
- Re-probe at K=2 and K=6 if first results are interesting.

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

### 9.2 Quantization mismatch can degrade acceptance

A drafter trained against the BF16 target sees slightly different
logits when paired with an NVFP4 target. Output remains
correct-by-construction (Sec. 6), but acceptance can drop several
percentage points. The right answer is to (re)train the drafter
on the *quantised* target's logits -- which is exactly what
production EAGLE3 recipes prescribe. Until then, expect Gemma 4
+ NVFP4 to show somewhat lower acceptance than Gemma 4 + BF16.

### 9.3 KV-cache pressure increases

The drafter does fewer FLOPs but consumes some KV slots. For
Gemma 4 the drafter *shares* the target's KV blocks, which is a
massive saving, but you still pay for the verifier's K extra
positions per round -- those K positions need K KV entries each
layer. For decode at 128K context this is sub-1 % of pool capacity;
for batched serving with many sequences in flight it can become a
real constraint on `--max-num-seqs`.

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

For Gemma 4 the drafter is ~150 MB - 900 MB BF16, ~0.5-2 s of
extra load time on this card -- a rounding error against the
~30-60 s NVFP4 cold-start path
([`nvfp4-coldstart.md`](nvfp4-coldstart.md) Sec. 2). For DeepSeek
V3 / Qwen3.6 built-in heads the drafter weights are already in
the target's checkpoint and there is no extra load step.

---

## 10. Practical recipe

Once the router changes from Sec. 7 land (or with the
`RecoveryFlags` minimum path), the fastest way to evaluate MTP on
this project:

```
    # 1. Make sure drafter is on disk next to the target
    ls /var/cache/devai/vllm/Gemma-4-26B-A4B-NVFP4
    ls /var/cache/devai/vllm/gemma-4-26B-A4B-it-assistant

    # 2. Add an entry to deploy/recovery-flags.json:
    #    {
    #      "Gemma-4-26B-A4B-NVFP4": {
    #        "engine_flags": [
    #          "--speculative-config",
    #          "{\"method\":\"mtp\",\"model\":\"/models/gemma-4-26B-A4B-it-assistant\",\"num_speculative_tokens\":4}"
    #        ]
    #      }
    #    }

    # 3. Re-probe to measure VRAM with the drafter loaded:
    make probe-vllm

    # 4. Launch the agent at a chosen context:
    devai-agent --model Gemma-4-26B-A4B-NVFP4@32768

    # 5. Compare decode tok/s against the same model WITHOUT
    #    recovery-flags entry (i.e. plain serving). Difference =
    #    MTP speedup on real workload.
```

For the built-in MTP case (Qwen3.6):

```
    # Same recipe, but no drafter directory needed -- the MTP head
    # is inside the same checkpoint:
    {
      "Qwen3.6-27B-Text-NVFP4-MTP": {
        "engine_flags": [
          "--speculative-config",
          "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":3}"
        ]
      }
    }
```

---

## 11. Summary

- Decode is bandwidth-bound; one forward pass produces one token,
  and the GPU spends most of its time reading weights.
- **MTP / speculative decoding** breaks the 1-token-per-pass
  barrier by drafting K tokens with a small fast drafter and
  verifying them in one big-model forward pass. Acceptance is
  guaranteed-correct by rejection sampling.
- Two architectures in 2026: **external drafter** (Gemma 4, EAGLE,
  Medusa) and **built-in MTP head** (DeepSeek V3, Qwen3.6). The
  verification math is identical; the difference is where the
  drafter's weights live.
- For this project's 24 GB RTX PRO 4000 Blackwell, the two prime
  candidates are **`nvidia/Gemma-4-26B-A4B-NVFP4` +
  `google/gemma-4-26B-A4B-it-assistant`** (lossless 2-3x speedup
  expected) and **`sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`**
  (self-contained, built-in MTP head). Both downloaded;
  `nvidia/Gemma-4-31B-IT-NVFP4` is the stretch candidate (tight
  on 24 GB at long context).
- The router does **not** currently support MTP. The cheapest path
  to evaluate is to drop the `--speculative-config` JSON into the
  existing `deploy/recovery-flags.json` per-model bag; the clean
  implementation adds a first-class `Speculative` field on
  `configModel` / `launchConfig`, an `::mtp` suffix override
  parser, and a probe-cache schema bump to record drafter VRAM
  overhead.
- "Lossless" is real: the output *distribution* is bit-identical
  to the unaccelerated model. Specific samples differ when
  temperature > 0 due to RNG ordering, but no quality is lost.

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
  stripping -> ctx injection) which an MTP-aware router would
  extend with an `::mtp`/`::nomtp` override parser.
- [`backends.md`](backends.md) -- backend lifecycle, where any
  `--speculative-config` flag must be injected before container
  start (a backend recreate is required to change the MTP
  configuration, same as for `currentModel` and `currentContext`).
- [`bench-results.md`](bench-results.md) -- the baseline tok/s
  numbers MTP would be compared against.
